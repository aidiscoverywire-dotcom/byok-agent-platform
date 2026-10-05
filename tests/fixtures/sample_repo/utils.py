"""Helpers."""


def load_config(raw: dict) -> dict:
    """Merge raw config over defaults."""
    defaults = {"debug": False, "port": 8000}
    return {**defaults, **raw}


def slugify(text: str) -> str:
    """Make a URL slug."""
    return text.lower().replace(" ", "-")
