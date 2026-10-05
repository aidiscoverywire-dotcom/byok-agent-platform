"""Fernet-encrypted key vault on any :class:`~byok_agent.db.base.Database` backend.

Plaintext provider keys exist only as Fernet tokens at rest. Decryption
happens inside :meth:`get_key` — the call boundary — and the plaintext is
handed to the caller for exactly one use.
"""
from __future__ import annotations

import secrets
from datetime import datetime, timezone

from cryptography.fernet import MultiFernet

from ..db.base import Database
from .crypto import load_multifernet
from .store import (
    ConcurrentModificationError,
    KeyNotFoundError,
    KeyRecord,
    KeyStatus,
    RevokedKeyError,
    SecretStore,
    VaultNotConfiguredError,
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_record(row: dict) -> KeyRecord:
    return KeyRecord(
        key_id=row["key_id"],
        label=row["label"],
        provider=row["provider"],
        encrypted_key=bytes(row["encrypted_key"]),
        key_version=int(row["key_version"]),
        status=KeyStatus(row["status"]),
        allow_cross_provider_fallback=bool(row["allow_cross_provider_fallback"]),
        owner=row.get("owner", "") or "",
        created_at=row["created_at"],
    )


class FernetVault:
    """A :class:`SecretStore` with Fernet column encryption.

    Args:
        db: Any :class:`~byok_agent.db.base.Database` (Postgres or SQLite).
        fernet: Optional pre-built MultiFernet (defaults to env-based).
    """

    def __init__(self, db: Database, fernet: MultiFernet | None = None) -> None:
        self._db = db
        self._fernet = fernet or load_multifernet()

    # -- SecretStore -----------------------------------------------------
    async def add_key(
        self,
        *,
        label: str,
        provider: str,
        plaintext_key: str,
        owner: str,
        allow_cross_provider_fallback: bool = False,
    ) -> KeyRecord:
        if not owner or not str(owner).strip():
            raise ValueError("owner is required: every key belongs to a user")
        # 128-bit random suffix — 32-bit was brute-forceable in bulk.
        key_id = f"key_{provider}_{secrets.token_hex(16)}"
        token = self._fernet.encrypt(plaintext_key.encode())
        record = KeyRecord(
            key_id=key_id,
            label=label,
            provider=provider,
            encrypted_key=token,
            owner=owner,
            allow_cross_provider_fallback=allow_cross_provider_fallback,
        )
        await self._db.execute(
            "INSERT INTO vault_keys (key_id, label, provider, owner, encrypted_key,"
            " key_version, status, allow_cross_provider_fallback, created_at)"
            " VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)",
            record.key_id,
            record.label,
            record.provider,
            record.owner,
            record.encrypted_key,
            record.key_version,
            record.status.value,
            int(record.allow_cross_provider_fallback),
            record.created_at,
        )
        return record

    async def get_record(
        self, key_id: str, owner: str | None = None
    ) -> KeyRecord:
        row = await self._db.fetchrow(
            "SELECT * FROM vault_keys WHERE key_id = $1", key_id
        )
        if row is None:
            raise KeyNotFoundError(f"unknown key_id {key_id!r}")
        record = _row_to_record(dict(row))
        if owner is not None and record.owner != owner:
            # No existence oracle: another tenant's key looks absent.
            raise KeyNotFoundError(f"unknown key_id {key_id!r}")
        return record

    async def get_key(self, key_id: str, owner: str | None = None) -> str:
        """Decrypt at the call boundary. Revoked keys fail closed."""
        record = await self.get_record(key_id, owner=owner)
        if record.status is not KeyStatus.ACTIVE:
            raise RevokedKeyError(
                f"key {key_id!r} is revoked and can never be used again"
            )
        try:
            return self._fernet.decrypt(record.encrypted_key).decode()
        except Exception as exc:
            # Wrong/rotated APP_VAULT_KEY (or corrupted row): a fatal but
            # ROUTINE misconfiguration — surface it cleanly, never as a
            # raw cryptography traceback.
            from cryptography.fernet import InvalidToken

            if isinstance(exc, InvalidToken):
                raise VaultNotConfiguredError(
                    "cannot decrypt stored keys — is APP_VAULT_KEY correct? "
                    "(If you rotated the master key, run rekey_vault with "
                    "APP_VAULT_KEY_PREVIOUS set.)"
                ) from exc
            raise

    async def list_keys(self, owner: str | None = None) -> list[KeyRecord]:
        if owner is None:
            rows = await self._db.fetch(
                "SELECT * FROM vault_keys ORDER BY created_at"
            )
        else:
            rows = await self._db.fetch(
                "SELECT * FROM vault_keys WHERE owner = $1 ORDER BY created_at",
                owner,
            )
        return [_row_to_record(dict(r)) for r in rows]

    async def rotate_key(self, key_id: str, new_plaintext_key: str) -> KeyRecord:
        record = await self.get_record(key_id)
        if record.status is not KeyStatus.ACTIVE:
            raise RevokedKeyError(f"cannot rotate revoked key {key_id!r}")
        token = self._fernet.encrypt(new_plaintext_key.encode())
        # Guarded UPDATE: only the rotation that saw THIS version wins.
        # A concurrent rotation bumps the version first, so this UPDATE
        # matches zero rows and we raise instead of overwriting it.
        await self._db.execute(
            "UPDATE vault_keys SET encrypted_key=$1, key_version=key_version+1"
            " WHERE key_id=$2 AND key_version=$3 AND status='active'",
            token,
            key_id,
            record.key_version,
        )
        after = await self.get_record(key_id)
        if after.key_version != record.key_version + 1:
            raise ConcurrentModificationError(
                f"key {key_id!r} was rotated concurrently; retry"
            )
        return after

    async def revoke_key(self, key_id: str) -> None:
        """Immediate revocation — no draining, no grace period."""
        await self.get_record(key_id)  # raises KeyNotFoundError if unknown
        await self._db.execute(
            "UPDATE vault_keys SET status='revoked' WHERE key_id=$1", key_id
        )

    async def rekey_vault(self) -> int:
        """Re-encrypt every row with the CURRENT master key.

        Run after rotating ``APP_VAULT_KEY`` (with the old key still in
        ``APP_VAULT_KEY_PREVIOUS``); afterwards the previous key can be
        dropped. Returns the number of rows re-encrypted.

        The UPDATE is guarded on ``(key_id, key_version, encrypted_key)``:
        if a rotation landed mid-rekey, the guard misses and we abort
        instead of re-encrypting (and thereby hiding) the new secret.
        """
        records = await self.list_keys()
        count = 0
        for record in records:
            plaintext = self._fernet.decrypt(record.encrypted_key)
            # encrypt() always uses the first (current) key of the MultiFernet
            fresh = self._fernet.encrypt(plaintext)
            await self._db.execute(
                "UPDATE vault_keys SET encrypted_key=$1"
                " WHERE key_id=$2 AND key_version=$3 AND encrypted_key=$4",
                fresh,
                record.key_id,
                record.key_version,
                record.encrypted_key,
            )
            after = await self.get_record(record.key_id)
            if after.key_version != record.key_version or bytes(
                after.encrypted_key
            ) != bytes(fresh):
                raise ConcurrentModificationError(
                    f"key {record.key_id!r} changed during rekey; aborting"
                )
            count += 1
        return count

    async def close(self) -> None:
        await self._db.close()
