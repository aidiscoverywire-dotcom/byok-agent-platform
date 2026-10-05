"""Minimal Python CLI starter (own code, MIT)."""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python-cli", description="minimal CLI starter")
    p.add_argument("--name", default="world", help="who to greet")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"hello, {args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
