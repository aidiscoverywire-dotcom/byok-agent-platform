"""Deploy targets: Vercel, Fly.io, Railway.

SECRET HANDLING (answers the audit question "deploy tokens kahan chalenge,
jab sandbox mein koi secret nahi?"):

* Deploy tokens live ONLY in the vault.
* At deploy time, the CLI resolves the needed token(s) from the vault ON THE
  HOST and injects them as environment variables into the deploy subprocess
  (``env={**os.environ, **secret_env}``).
* Secrets NEVER enter the sandbox, NEVER get written to files, and NEVER
  appear in logs — every log line goes through
  :func:`~byok_agent.llm.sanitize_secret_text`.

Production deploys (``--prod``) require human approval (``interrupt()`` /
dashboard approval queue). ``--auto-approve`` exists for CI but the CLI
prints a loud warning when it is used.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from byok_agent.llm import sanitize_secret_text

log = logging.getLogger(__name__)


class DeployApprovalNeeded(Exception):
    """A production deploy was attempted without human approval."""


@dataclass
class DeployResult:
    ok: bool
    platform: str
    output: str  # sanitized log tail
    url: str | None = None


# Runner is injectable so tests never spawn real CLIs.
Runner = Callable[..., "CompletedLike"]


@dataclass
class CompletedLike:
    returncode: int
    stdout: str = ""
    stderr: str = ""


def _default_runner(cmd: list[str], **kwargs: Any) -> CompletedLike:
    proc = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    return CompletedLike(
        returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr
    )


@runtime_checkable
class DeployBackend(Protocol):
    """One deploy target."""

    name: str

    def detect(self, project_dir: Path) -> bool:
        """True if this backend looks right for the project."""
        ...

    def prechecks(self, project_dir: Path) -> list[str]:
        """Return a list of problems (empty = good to go)."""
        ...

    async def deploy(
        self,
        project_dir: Path,
        *,
        env_extra: dict[str, str],
        prod: bool,
        approved: bool,
        runner: Runner = _default_runner,
    ) -> DeployResult:
        ...


class _BaseBackend:
    """Shared CLI-existence precheck + sanitized subprocess runner."""

    cli_name = ""
    prod_flag: list[str] = field(default_factory=list)  # type: ignore[assignment]

    @property
    def name(self) -> str:  # type: ignore[override]
        return self.cli_name

    def _cli_problems(self) -> list[str]:
        if not self.cli_name or shutil.which(self.cli_name) is None:
            return [f"CLI not found on PATH: {self.cli_name!r}"]
        return []

    def _run_deploy(
        self,
        cmd: list[str],
        project_dir: Path,
        env_extra: dict[str, str],
        runner: Runner,
    ) -> DeployResult:
        env = {**os.environ, **env_extra}
        log.info(
            "deploying via: %s",
            sanitize_secret_text(" ".join(cmd)),
        )
        completed = runner(cmd, cwd=str(project_dir), env=env)
        tail = sanitize_secret_text(
            (completed.stdout + "\n" + completed.stderr)[-4000:]
        )
        url = _extract_url(tail)
        return DeployResult(
            ok=completed.returncode == 0,
            platform=self.name,
            output=tail,
            url=url,
        )


def _extract_url(text: str) -> str | None:
    import re

    m = re.search(r"https://[a-zA-Z0-9][\w.\-/]*", text)
    return m.group(0) if m else None


class VercelDeploy(_BaseBackend):
    """Vercel: Next.js / static frontends."""

    cli_name = "vercel"

    def detect(self, project_dir: Path) -> bool:
        if (project_dir / "vercel.json").exists():
            return True
        pkg = project_dir / "package.json"
        return pkg.exists() and "next" in pkg.read_text(errors="ignore")

    def prechecks(self, project_dir: Path) -> list[str]:
        problems = self._cli_problems()
        if not (project_dir / "package.json").exists():
            problems.append("no package.json in project dir")
        return problems

    async def deploy(
        self, project_dir: Path, *, env_extra: dict[str, str], prod: bool,
        approved: bool, runner: Runner = _default_runner,
    ) -> DeployResult:
        if prod and not approved:
            raise DeployApprovalNeeded(
                "production Vercel deploy needs human approval"
            )
        cmd = ["vercel", "--yes"] + (["--prod"] if prod else [])
        return self._run_deploy(cmd, project_dir, env_extra, runner)


class FlyDeploy(_BaseBackend):
    """Fly.io: containers / backends (fly.toml)."""

    cli_name = "fly"

    def detect(self, project_dir: Path) -> bool:
        return (project_dir / "fly.toml").exists()

    def prechecks(self, project_dir: Path) -> list[str]:
        problems = self._cli_problems()
        if not (project_dir / "fly.toml").exists():
            problems.append("no fly.toml in project dir")
        if not (project_dir / "Dockerfile").exists():
            problems.append("no Dockerfile in project dir (fly needs one)")
        return problems

    async def deploy(
        self, project_dir: Path, *, env_extra: dict[str, str], prod: bool,
        approved: bool, runner: Runner = _default_runner,
    ) -> DeployResult:
        # Fly has no staging/prod split in one command; treat every deploy
        # as production-grade -> always needs approval unless auto-approved.
        if not approved:
            raise DeployApprovalNeeded(
                "Fly deploy needs human approval (use --auto-approve in CI)"
            )
        return self._run_deploy(["fly", "deploy"], project_dir, env_extra, runner)


class RailwayDeploy(_BaseBackend):
    """Railway: `railway up`."""

    cli_name = "railway"

    def detect(self, project_dir: Path) -> bool:
        return (
            (project_dir / "railway.json").exists()
            or (project_dir / "railway.toml").exists()
        )

    def prechecks(self, project_dir: Path) -> list[str]:
        return self._cli_problems()

    async def deploy(
        self, project_dir: Path, *, env_extra: dict[str, str], prod: bool,
        approved: bool, runner: Runner = _default_runner,
    ) -> DeployResult:
        if not approved:
            raise DeployApprovalNeeded(
                "Railway deploy needs human approval (use --auto-approve in CI)"
            )
        return self._run_deploy(["railway", "up"], project_dir, env_extra, runner)


BACKENDS: list[DeployBackend] = [VercelDeploy(), FlyDeploy(), RailwayDeploy()]


def autodetect(project_dir: Path) -> DeployBackend | None:
    """First backend whose detect() matches, else None."""
    for backend in BACKENDS:
        if backend.detect(project_dir):
            return backend
    return None


def get_backend(name: str) -> DeployBackend:
    for backend in BACKENDS:
        if backend.name == name:
            return backend
    raise KeyError(f"unknown deploy platform: {name!r}")
