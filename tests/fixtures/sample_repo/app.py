"""Application factory."""
from .models import User, Session
from . import utils


def create_app(config: dict | None = None) -> dict:
    """Build the app object."""
    settings = utils.load_config(config or {})
    return {"settings": settings, "models": [User, Session]}


def run(port: int = 8000) -> None:
    """Serve the app (demo only)."""
    app = create_app()
    print(f"serving on {port}: {app}")


async def health_check() -> dict:
    """Liveness probe payload."""
    return {"status": "ok"}
