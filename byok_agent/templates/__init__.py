"""App starter templates (all original, MIT-licensed — no third-party license risk)."""
from __future__ import annotations

import shutil
from pathlib import Path

_TEMPLATES_DIR = Path(__file__).parent

DESCRIPTIONS = {
    "fastapi-min": "Minimal FastAPI app (health endpoint + Dockerfile)",
    "next-min": "Minimal Next.js app router starter (health API route)",
    "python-cli": "Minimal Python CLI (argparse + pytest)",
}


def list_templates() -> list[dict]:
    """Available starters: [{'name', 'description'}]."""
    return [
        {"name": name, "description": DESCRIPTIONS[name]}
        for name in sorted(DESCRIPTIONS)
        if (_TEMPLATES_DIR / name).is_dir()
    ]


def scaffold(name: str, dest: Path) -> Path:
    """Copy a starter into ``dest`` (created if missing). Returns dest."""
    src = _TEMPLATES_DIR / name
    if not src.is_dir() or name not in DESCRIPTIONS:
        raise KeyError(f"unknown template: {name!r}")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
    return dest
