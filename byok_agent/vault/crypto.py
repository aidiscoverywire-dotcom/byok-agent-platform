"""Fernet crypto helpers with MultiFernet master-key rotation.

Master key material comes from the environment::

    APP_VAULT_KEY             current Fernet key (base64 urlsafe, 32 bytes)
    APP_VAULT_KEY_PREVIOUS    previous key, kept only during rotation

Encryption always uses the FIRST (current) key; decryption tries each key
in order, so rows encrypted under the previous key keep working until the
operator runs :meth:`FernetVault.rekey_vault` and drops the old key.
"""
from __future__ import annotations

import os

from cryptography.fernet import Fernet, MultiFernet

from .store import VaultNotConfiguredError


def load_multifernet() -> MultiFernet:
    """Build a MultiFernet from the environment.

    Raises :class:`VaultNotConfiguredError` if ``APP_VAULT_KEY`` is missing
    or not a valid Fernet key.
    """
    current = os.environ.get("APP_VAULT_KEY", "").strip()
    if not current:
        raise VaultNotConfiguredError(
            "APP_VAULT_KEY is not set. Generate one with:\n"
            "  python -c \"from cryptography.fernet import Fernet; "
            "print(Fernet.generate_key().decode())\""
        )
    try:
        keys = [Fernet(current.encode())]
    except Exception as exc:
        raise VaultNotConfiguredError(
            f"APP_VAULT_KEY is not a valid Fernet key: {exc}"
        ) from exc

    previous = os.environ.get("APP_VAULT_KEY_PREVIOUS", "").strip()
    if previous:
        try:
            keys.append(Fernet(previous.encode()))
        except Exception as exc:
            raise VaultNotConfiguredError(
                f"APP_VAULT_KEY_PREVIOUS is not a valid Fernet key: {exc}"
            ) from exc
    return MultiFernet(keys)


def generate_master_key() -> str:
    """Generate a fresh Fernet master key (for ``APP_VAULT_KEY``)."""
    return Fernet.generate_key().decode()
