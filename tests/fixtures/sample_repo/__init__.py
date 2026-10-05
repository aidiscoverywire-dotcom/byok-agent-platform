"""Sample repo for codemap tests: a tiny web app with cross-file refs."""
from .app import create_app
from . import models

__all__ = ["create_app", "models"]
