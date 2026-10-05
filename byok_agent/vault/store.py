"""Key-vault contracts: envelope shape, statuses, and the SecretStore protocol."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol, runtime_checkable


class KeyStatus(str, Enum):
    ACTIVE = "active"
    REVOKED = "revoked"


class VaultError(Exception):
    """Base class for vault failures."""


class KeyNotFoundError(VaultError):
    """No key with this id exists."""


class RevokedKeyError(VaultError):
    """The key was revoked. Revoked keys are NEVER usable again."""


class VaultNotConfiguredError(VaultError):
    """Master key material is missing or malformed."""


class ConcurrentModificationError(VaultError):
    """A key changed between read and write — retry the operation."""


@dataclass
class KeyRecord:
    """Metadata envelope for one provider API key.

    The plaintext key is NEVER stored — only the Fernet token in
    ``encrypted_key``. Plaintext exists only transiently at the call
    boundary (see :meth:`SecretStore.get_key`).

    ``owner`` is the tenant scope: a user may only see or use keys they
    own. Reads scoped with an ``owner`` that doesn't match behave as if
    the key doesn't exist (no existence oracle for other tenants).
    """

    key_id: str
    label: str
    provider: str
    encrypted_key: bytes
    key_version: int = 1
    status: KeyStatus = KeyStatus.ACTIVE
    allow_cross_provider_fallback: bool = False
    owner: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@runtime_checkable
class SecretStore(Protocol):
    """Storage contract for BYOK provider keys."""

    async def add_key(
        self,
        *,
        label: str,
        provider: str,
        plaintext_key: str,
        owner: str,
        allow_cross_provider_fallback: bool = False,
    ) -> KeyRecord:
        """Encrypt and store a new key. Returns the envelope (no secret)."""
        ...

    async def get_record(self, key_id: str, owner: str | None = None) -> KeyRecord:
        """Return the envelope (metadata only, never the secret).

        When ``owner`` is given, a key owned by someone else behaves as if
        it doesn't exist (:class:`KeyNotFoundError`).
        """
        ...

    async def get_key(self, key_id: str, owner: str | None = None) -> str:
        """Decrypt and return the plaintext key — CALL BOUNDARY ONLY.

        Raises :class:`KeyNotFoundError` / :class:`RevokedKeyError`.
        Callers must use the value immediately for one provider call and
        must not cache, log, or persist it.
        """
        ...

    async def list_keys(self, owner: str | None = None) -> list[KeyRecord]:
        """List envelopes (metadata only), optionally scoped to one owner."""
        ...

    async def rotate_key(self, key_id: str, new_plaintext_key: str) -> KeyRecord:
        """Replace the stored secret, bumping ``key_version``.

        The UPDATE is guarded on ``(key_id, key_version, status='active')``:
        a concurrent rotation wins exactly once; the loser gets
        :class:`ConcurrentModificationError` instead of silently
        overwriting the winner's secret.
        """
        ...

    async def revoke_key(self, key_id: str) -> None:
        """Revoke IMMEDIATELY. No draining: a revoked key fails closed on
        the very next use."""
        ...

    async def close(self) -> None:
        """Release underlying resources."""
        ...
