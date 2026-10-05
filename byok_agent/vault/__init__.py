"""Encrypted BYOK key vault: Fernet at rest, MultiFernet rotation, instant revoke."""
from .crypto import generate_master_key, load_multifernet
from .fernet_vault import FernetVault
from .sqlite_vault import SqliteVault
from .store import (
    ConcurrentModificationError,
    KeyNotFoundError,
    KeyRecord,
    KeyStatus,
    RevokedKeyError,
    SecretStore,
    VaultError,
    VaultNotConfiguredError,
)

__all__ = [
    "generate_master_key",
    "load_multifernet",
    "FernetVault",
    "SqliteVault",
    "ConcurrentModificationError",
    "KeyNotFoundError",
    "KeyRecord",
    "KeyStatus",
    "RevokedKeyError",
    "SecretStore",
    "VaultError",
    "VaultNotConfiguredError",
]
