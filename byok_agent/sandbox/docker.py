"""Docker-backed sandbox: host-driven, no in-guest listeners.

Lifecycle (host side only):

1. ``start()`` creates ONE long-lived container per task
   (``sleep infinity`` as an unprivileged user) with a host bind-mounted
   workdir at ``/work``.
2. ``copy_in`` / ``write_file`` stage files into the host workdir — they
   are instantly visible inside the container (real directory, not tmpfs,
   so files survive across exec calls).
3. ``exec`` runs commands via the Docker API (``docker exec`` over the
   socket — works fine with ``--network none`` since no guest networking
   is involved).
4. ``stop()`` removes the container (force) and optionally the workdir.

Security defaults (deny by default):

- ``network_mode="none"`` — no egress at all.
- ``mem_limit="2g"``, ``nano_cpus=2`` (2 CPUs), ``pids_limit=256``.
- ``user="65534:65534"`` (nobody), ``cap_drop=["ALL"]``,
  ``security_opt=["no-new-privileges"]``.
- ``read_only=True`` root filesystem; writable ``/tmp`` via tmpfs and
  ``/work`` via the bind mount only.

Network honesty: with ``--network none``, package managers (pip/npm/
Maven/Cargo) CANNOT fetch dependencies. Either use images with pre-baked
deps, or pass ``install_network=<name>`` — the container then attaches to
an operator-provided egress-filtered network (e.g. proxy-only) instead of
``none``. The default stays deny.

Budget integration: ``exec`` consults the Phase 0
:class:`~byok_agent.agent.budgets.BudgetEnforcer` BEFORE starting, and the
wait loop polls the kill switch (plus an optional ``abort_event``); on
either, the running container is killed immediately.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from .backend import (
    SandboxAbortedError,
    SandboxDeadError,
    SandboxError,
    SandboxResult,
    _check_relpath,
    run_with_abort,
)

log = logging.getLogger(__name__)

LANGUAGE_IMAGES: dict[str, str] = {
    "python": "python:3.12-slim",
    "node": "node:20-slim",
    "java": "maven:3-eclipse-temurin-17",
    "go": "golang:1.22",
    "rust": "rust:1.75-slim",
}

#: Test commands used by the builder's tester node when the plan doesn't
#: specify one.
DEFAULT_TEST_COMMANDS: dict[str, str] = {
    "python": "python -m pytest -q",
    "node": "npm test --silent",
    "java": "mvn -q -o test",  # offline: pre-baked repo only
    "go": "go test ./...",
    "rust": "cargo test --offline --quiet",
}

_NOBODY = "65534:65534"


def is_docker_available() -> bool:
    """True if a Docker daemon is reachable from this host."""
    try:
        import docker

        client = docker.from_env()
        client.ping()
        client.close()
        return True
    except Exception as exc:  # daemon down, no socket, no perms
        log.debug("docker unavailable: %s", exc)
        return False


class DockerSandbox:
    """A per-task Docker container, driven host-side via the Docker API.

    Args:
        language: key into :data:`LANGUAGE_IMAGES`.
        enforcer: Phase 0 :class:`~byok_agent.agent.budgets.BudgetEnforcer`;
            consulted before every exec (kill switch + step caps).
        install_network: if set, the container attaches to this named
            network INSTEAD of ``none`` (operator-provided egress-filtered
            network for dependency installs). Default ``None`` = deny.
        keep_workdir: keep the host workdir after :meth:`stop` (debugging).
        image: override the language image (e.g. a pre-baked variant).
    """

    def __init__(
        self,
        language: str = "python",
        *,
        enforcer: Any = None,
        install_network: str | None = None,
        keep_workdir: bool = False,
        image: str | None = None,
    ) -> None:
        if language not in LANGUAGE_IMAGES and image is None:
            raise SandboxError(
                f"unknown language {language!r}; choose from "
                f"{sorted(LANGUAGE_IMAGES)} or pass image="
            )
        self._language = language
        self._image = image or LANGUAGE_IMAGES[language]
        self._enforcer = enforcer
        self._install_network = install_network
        self._keep_workdir = keep_workdir
        self._task_id = uuid.uuid4().hex[:12]
        self._workdir = Path(tempfile.mkdtemp(prefix=f"byok-sbx-{self._task_id}-"))
        # The container runs as nobody (65534) but the host workdir is
        # mkdtemp'd 0700 owned by the invoking user — without this chmod the
        # guest cannot write to /work at all and every build fails on first
        # real Docker run. The dir is ephemeral (removed on stop()).
        os.chmod(self._workdir, 0o777)
        self._client: Any = None
        self._container: Any = None
        self._dead = False

    @property
    def workdir(self) -> Path:
        return self._workdir

    @property
    def language(self) -> str:
        return self._language

    def _require_live(self) -> None:
        if self._dead:
            raise SandboxDeadError(
                "sandbox was killed (timeout/abort) and cannot be reused; "
                "start a fresh one"
            )
        if self._container is None:
            raise SandboxError("sandbox not started; call start() first")

    # -- lifecycle ------------------------------------------------------
    async def start(self) -> None:
        """Create the task container (idempotent)."""
        if self._container is not None:
            return
        try:
            import docker
        except ImportError as exc:
            raise SandboxError(
                "the 'docker' package is not installed; "
                "pip install -r requirements.txt"
            ) from exc
        network_mode = self._install_network or "none"
        if self._install_network:
            log.warning(
                "sandbox %s: egress ENABLED via network %r (install mode); "
                "default-deny is off for this task",
                self._task_id,
                self._install_network,
            )
        try:
            self._client = await asyncio.to_thread(docker.from_env)
            await asyncio.to_thread(self._client.ping)
            self._container = await asyncio.to_thread(
                self._client.containers.run,
                self._image,
                ["sleep", "infinity"],
                detach=True,
                name=f"byok-sandbox-{self._task_id}",
                network_mode=network_mode,
                mem_limit="2g",
                nano_cpus=2_000_000_000,
                pids_limit=256,
                user=_NOBODY,
                read_only=True,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges"],
                tmpfs={"/tmp": "size=64m,mode=1777"},
                volumes={str(self._workdir): {"bind": "/work", "mode": "rw"}},
                working_dir="/work",
            )
        except Exception as exc:
            raise SandboxError(f"could not start sandbox container: {exc}") from exc
        log.info(
            "sandbox %s started: image=%s network=%s workdir=%s",
            self._task_id,
            self._image,
            network_mode,
            self._workdir,
        )

    async def stop(self) -> None:
        """Remove the container (force) and clean up the workdir."""
        if self._container is not None:
            try:
                await asyncio.to_thread(self._container.remove, force=True)
            except Exception as exc:  # already gone / daemon hiccup
                log.debug("container remove failed (ignored): %s", exc)
            self._container = None
        if self._client is not None:
            try:
                await asyncio.to_thread(self._client.close)
            except Exception:
                pass
            self._client = None
        if not self._keep_workdir:
            shutil.rmtree(self._workdir, ignore_errors=True)
        log.info("sandbox %s stopped", self._task_id)

    async def _kill_guest(self) -> None:
        """Kill the running container; the sandbox is dead afterwards."""
        self._dead = True
        if self._container is not None:
            try:
                await asyncio.to_thread(self._container.kill)
            except Exception as exc:
                log.debug("container kill failed (ignored): %s", exc)

    # -- file movement (host-side; bind mount makes it instant) ---------
    async def write_file(self, path: str, content: str) -> None:
        rel = _check_relpath(path)
        target = self._workdir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(target.write_text, content, encoding="utf-8")

    async def read_file(self, path: str) -> str:
        rel = _check_relpath(path)
        target = self._workdir / rel
        if not target.is_file():
            raise SandboxError(f"no such file in sandbox: {rel!r}")
        return await asyncio.to_thread(target.read_text, encoding="utf-8")

    async def copy_in(self, local_dir: str | Path) -> None:
        """Copy a whole local tree into the workdir (task setup)."""
        src = Path(local_dir)
        if not src.is_dir():
            raise SandboxError(f"copy_in source is not a directory: {src}")
        def _copy() -> None:
            for item in src.rglob("*"):
                rel = item.relative_to(src)
                # Skip dotfiles/dirs at ANY depth (.git, .venv, .env, ...):
                # code only. (The old check only skipped top-level dot
                # entries, so nested secrets like `sub/.env` leaked into the
                # sandbox — contradicting the no-secrets-in-sandbox rule.)
                if any(part.startswith(".") for part in rel.parts):
                    continue
                dest = self._workdir / rel
                if item.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                elif item.is_file():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dest)
        await asyncio.to_thread(_copy)
        log.info("sandbox %s: copied in %s", self._task_id, src)

    async def copy_out(
        self, local_dir: str | Path, paths: list[str] | None = None
    ) -> None:
        """Copy workdir files out to a local tree (promotion)."""
        dest_root = Path(local_dir)
        def _copy() -> None:
            if paths is None:
                items = [p for p in self._workdir.rglob("*") if p.is_file()]
            else:
                items = [self._workdir / _check_relpath(p) for p in paths]
            for src in items:
                if not src.is_file():
                    continue
                rel = src.relative_to(self._workdir)
                dest = dest_root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
        await asyncio.to_thread(_copy)

    # -- execution --------------------------------------------------------
    async def exec(
        self,
        cmd: str | list[str],
        timeout_s: float = 120.0,
        *,
        abort_event: asyncio.Event | None = None,
    ) -> SandboxResult:
        """Run a command via the Docker API with timeout + abort supervision.

        Budget gate runs BEFORE the exec starts. While waiting, the loop
        polls the kill switch and ``abort_event``; either kills the running
        container immediately (the sandbox is then dead).
        """
        self._require_live()
        assert self._container is not None

        # Budget gate BEFORE starting anything (also enforces kill switch).
        if self._enforcer is not None:
            await self._enforcer.before_tool_call("sandbox_exec")

        from byok_agent.agent.budgets import kill_switch_engaged

        if isinstance(cmd, str):
            argv: str | list[str] = ["/bin/sh", "-c", cmd]
        else:
            argv = list(cmd)

        def _do_exec() -> SandboxResult:
            try:
                res = self._container.exec_run(
                    argv, workdir="/work", user=_NOBODY, demux=True
                )
            except Exception as exc:
                return SandboxResult(exit_code=125, stderr=f"exec failed: {exc}")
            out = res.output if isinstance(res.output, tuple) else (res.output, b"")
            stdout_b, stderr_b = out[0] or b"", out[1] or b""
            return SandboxResult(
                exit_code=res.exit_code,
                stdout=stdout_b.decode("utf-8", "replace"),
                stderr=stderr_b.decode("utf-8", "replace"),
            )

        async def _factory() -> SandboxResult:
            return await asyncio.to_thread(_do_exec)

        async def _on_abort() -> None:
            await self._kill_guest()

        # One abort signal for both sources: the caller's event (budget
        # abort) and the kill switch (polled by the watchdog below). Either
        # one funnels through run_with_abort -> _kill_guest, so the running
        # container is always killed and the sandbox marked dead.
        local_abort = abort_event if abort_event is not None else asyncio.Event()
        stop_watch = asyncio.Event()

        async def _watchdog() -> None:
            while not stop_watch.is_set():
                if kill_switch_engaged():
                    local_abort.set()
                    return
                await asyncio.sleep(0.2)

        watch_task = asyncio.ensure_future(_watchdog())
        try:
            return await run_with_abort(
                _factory,
                timeout_s=timeout_s,
                abort_event=local_abort,
                on_abort=_on_abort,
                abort_label="sandbox exec aborted (kill switch / budget abort)",
            )
        finally:
            stop_watch.set()
            try:
                await asyncio.wait_for(watch_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                watch_task.cancel()
