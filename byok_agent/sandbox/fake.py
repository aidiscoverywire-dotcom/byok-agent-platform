"""In-memory fake sandbox: same protocol, no Docker required.

Used by the test suite and as a documented fallback for environments
without a Docker daemon. Command handling is scriptable: pass
``handlers`` mapping a substring to either a canned
:class:`~byok_agent.sandbox.backend.SandboxResult` or a callable
``(cmd, sandbox) -> SandboxResult`` (sync or async).

Abort/timeout semantics mirror :class:`DockerSandbox` via
:func:`~byok_agent.sandbox.backend.run_with_abort` — a killed fake is
likewise dead afterwards.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any, Awaitable, Callable

from .backend import (
    SandboxAbortedError,
    SandboxDeadError,
    SandboxError,
    SandboxResult,
    _check_relpath,
    run_with_abort,
)

log = logging.getLogger(__name__)

Handler = Callable[[str, "FakeSandbox"], Any]


class FakeSandbox:
    """In-memory :class:`~byok_agent.sandbox.backend.SandboxBackend`.

    Args:
        handlers: substring -> SandboxResult | (sync/async) callable.
            The first matching substring wins; no match returns
            ``exit_code=0`` with empty output.
        enforcer: optional Phase 0 BudgetEnforcer, consulted before exec.
        workdir: optional host dir to use instead of an internal dict.
            When set, files live on disk (lets ``copy_in`` read real
            fixture trees and ``copy_out`` land real files).
    """

    def __init__(
        self,
        handlers: dict[str, Any] | None = None,
        *,
        enforcer: Any = None,
        workdir: str | Path | None = None,
    ) -> None:
        self._handlers = dict(handlers or {})
        self._enforcer = enforcer
        self._files: dict[str, str] = {}
        self._disk_workdir = Path(workdir) if workdir is not None else None
        if self._disk_workdir is not None:
            self._disk_workdir.mkdir(parents=True, exist_ok=True)
        self._started = False
        self._dead = False
        self.exec_calls: list[str] = []
        self.killed = False

    @property
    def workdir(self) -> Path:
        if self._disk_workdir is not None:
            return self._disk_workdir
        # Synthetic path for dict mode (never touched on disk).
        return Path("/fake-sandbox-workdir")

    def _require_live(self) -> None:
        if self._dead:
            raise SandboxDeadError("fake sandbox was killed; start a fresh one")
        if not self._started:
            raise SandboxError("sandbox not started; call start() first")

    async def start(self) -> None:
        self._started = True
        self._dead = False
        log.debug("FakeSandbox started")

    async def stop(self) -> None:
        self._started = False
        log.debug("FakeSandbox stopped")

    async def _kill_guest(self) -> None:
        self._dead = True
        self.killed = True

    # -- files ----------------------------------------------------------
    def _put(self, rel: str, content: str) -> None:
        if self._disk_workdir is not None:
            target = self._disk_workdir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        else:
            self._files[rel] = content

    def _get(self, rel: str) -> str:
        if self._disk_workdir is not None:
            target = self._disk_workdir / rel
            if not target.is_file():
                raise SandboxError(f"no such file in sandbox: {rel!r}")
            return target.read_text(encoding="utf-8")
        if rel not in self._files:
            raise SandboxError(f"no such file in sandbox: {rel!r}")
        return self._files[rel]

    async def write_file(self, path: str, content: str) -> None:
        self._require_live()
        self._put(_check_relpath(path), content)

    async def read_file(self, path: str) -> str:
        self._require_live()
        return self._get(_check_relpath(path))

    async def copy_in(self, local_dir: str | Path) -> None:
        self._require_live()
        src = Path(local_dir)
        if not src.is_dir():
            raise SandboxError(f"copy_in source is not a directory: {src}")
        for item in src.rglob("*"):
            if not item.is_file():
                continue
            rel = item.relative_to(src)
            if rel.parts and rel.parts[0].startswith("."):
                continue  # skip dotfiles/dirs (.git, .venv, ...) — code only
            self._put(rel.as_posix(), item.read_text(encoding="utf-8"))

    async def copy_out(
        self, local_dir: str | Path, paths: list[str] | None = None
    ) -> None:
        self._require_live()
        dest_root = Path(local_dir)
        if self._disk_workdir is not None:
            names = (
                paths
                if paths is not None
                else [
                    p.relative_to(self._disk_workdir).as_posix()
                    for p in self._disk_workdir.rglob("*")
                    if p.is_file()
                ]
            )
            for name in names:
                rel = _check_relpath(name)
                src = self._disk_workdir / rel
                if not src.is_file():
                    continue
                dest = dest_root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
            return
        names = paths if paths is not None else list(self._files)
        for name in names:
            rel = _check_relpath(name)
            if rel not in self._files:
                continue
            dest = dest_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(self._files[rel], encoding="utf-8")

    # -- execution --------------------------------------------------------
    def _dispatch(self, cmd: str) -> Any:
        for pattern, handler in self._handlers.items():
            if pattern in cmd:
                return handler
        return None

    async def exec(
        self,
        cmd: str | list[str],
        timeout_s: float = 120.0,
        *,
        abort_event: asyncio.Event | None = None,
    ) -> SandboxResult:
        self._require_live()
        if self._enforcer is not None:
            await self._enforcer.before_tool_call("sandbox_exec")

        cmd_s = cmd if isinstance(cmd, str) else " ".join(cmd)
        self.exec_calls.append(cmd_s)
        handler = self._dispatch(cmd_s)

        async def _factory() -> SandboxResult:
            if handler is None:
                return SandboxResult(exit_code=0, stdout="", stderr="")
            if isinstance(handler, SandboxResult):
                return handler
            res = handler(cmd_s, self)
            if asyncio.iscoroutine(res):
                res = await res
            if isinstance(res, SandboxResult):
                return res
            # Bare string -> stdout shortcut.
            return SandboxResult(exit_code=0, stdout=str(res))

        async def _on_abort() -> None:
            await self._kill_guest()

        return await run_with_abort(
            _factory,
            timeout_s=timeout_s,
            abort_event=abort_event,
            on_abort=_on_abort,
            abort_label="fake sandbox exec aborted",
        )
