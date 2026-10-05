"""Deploy targets."""
from .backends import (
    BACKENDS,
    DeployApprovalNeeded,
    DeployBackend,
    DeployResult,
    FlyDeploy,
    RailwayDeploy,
    VercelDeploy,
    autodetect,
    get_backend,
)

__all__ = [
    "BACKENDS",
    "DeployApprovalNeeded",
    "DeployBackend",
    "DeployResult",
    "FlyDeploy",
    "RailwayDeploy",
    "VercelDeploy",
    "autodetect",
    "get_backend",
]
