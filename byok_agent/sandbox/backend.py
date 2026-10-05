"""The sandbox backend protocol and shared execution helpers.

Design notes (from the audited threat model):

- There is NO agent/listener *inside* the container. The host drives the
  container exclusively through the Docker API (``docker exec`` over the
  socket). A container with ``--network none`` can therefore still be
  driven — the contradiction of an "in-container HTTP supervisor" behind
  a no-network default is gone.
- Files move by copy-in/copy-out across a host-side bind-mounted workdir.
  The workdir is a REAL directory on the host (not tmpfs), so files and
  installed dependencies survive across ``exec`` calls within one task.
  The container is destroyed when the task ends or times out.
- With ``--network none`` (the default), ``npm``/``pip``/Maven/Cargo
  CANNOT fetch dependencies at build time. Language images must pre-bake
  common deps, or the operator enables ``install_network`` (an
  egress-filtered network) explicitly per task. Default is deny.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

log = logging.getLogger(__name__)


class SandboxError(Exception):
    """Base class for sandbox failures."""


class SandboxTimeoutError(SandboxError):
    """An exec exceeded its timeout; the container was killed."""


class SandboxAbortedError(SandboxError):
    """An exec was aborted (kill switch / budget abort); container killed."""


class SandboxDeadError(SandboxError):
    """The sandbox was killed (timeout/abort) and cannot be reused."""


@dataclass(frozen=True)
class SandboxResult:
    """Result of one sandboxed command."""

    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


@runtime_checkable
class SandboxBackend(Protocol):
    """Host-driven sandbox contract.

    Implementations MUST be host-driven: no listeners inside the guest.
    ``copy_in``/``copy_out`` move whole trees; ``write_file``/``read_file``
    move single files. Paths are always relative to the task workdir
    (``..`` and absolute paths are rejected).
    """

    async def start(self) -> None:
        """Create the sandbox (container, VM, etc.)."""
        ...

    async def exec(
        self,
        cmd: str | list[str],
        timeout_s: float = 120.0,
        *,
        abort_event: asyncio.Event | None = None,
    ) -> SandboxResult:
        """Run a command inside the sandbox.

        Must consult the budget gate BEFORE starting, and must kill the
        running command when ``abort_event`` is set or the timeout fires.
        """
        ...

    async def write_file(self, path: str, content: str) -> None:
        """Write one file into the task workdir."""
        ...

    async def read_file(self, path: str) -> str:
        """Read one file from the task workdir."""
        ...

    async def copy_in(self, local_dir: str | Path) -> None:
        """Copy a whole local tree INTO the task workdir."""
        ...

    async def copy_out(self, local_dir: str | Path, paths: list[str] | None = None) -> None:
        """Copy files OUT of the task workdir into a local tree."""
        ...

    async def stop(self) -> None:
        """Destroy the sandbox and release resources."""
        ...

    @property
    def workdir(self) -> Path:
        """Host-side path of the task workdir (copy-in/out staging area)."""
        ...


def _check_relpath(path: str) -> str:
    """Reject absolute paths and ``..`` escapes; return normalized relpath."""
    p = Path(path)
    if p.is_absolute() or ".." in p.parts:
        raise SandboxError(f"refusing unsafe sandbox path: {path!r}")
    return p.as_posix()


async def run_with_abort(
    awaitable_factory: Callable[[], Awaitable[SandboxResult]],
    *,
    timeout_s: float,
    abort_event: asyncio.Event | None,
    on_abort: Callable[[], Awaitable[None]],
    abort_label: str = "aborted",
) -> SandboxResult:
    """Await a sandbox exec with timeout + abort supervision.

    Shared by backends so timeout/abort semantics are identical:

    - ``timeout_s`` elapses → ``on_abort`` (kill the guest), raise
      :class:`SandboxTimeoutError`.
    - ``abort_event`` set (kill switch / budget abort) → ``on_abort``,
      raise :class:`SandboxAbortedError`.
    - Normal completion → return the result.
    """
    task = asyncio.ensure_future(awaitable_factory())

    async def _supervised() -> SandboxResult:
        while not task.done():
            if abort_event is not None and abort_event.is_set():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
                await on_abort()
                raise SandboxAbortedError(abort_label)
            # Poll at 10 Hz — responsive enough for aborts, cheap enough
            # to not matter next to a container exec.
            try:
                return await asyncio.wait_for(asyncio.shield(task), timeout=0.1)
            except asyncio.TimeoutError:
                continue
        return task.result()

    try:
        return await asyncio.wait_for(_supervised(), timeout=timeout_s)
    except asyncio.TimeoutError:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        await on_abort()
        raise SandboxTimeoutError(f"exec exceeded {timeout_s:.1f}s; guest killed")
