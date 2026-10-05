"""Sandboxed code execution backends.

The :class:`SandboxBackend` protocol is the seam where managed sandboxes
(E2B, etc.) plug in later: anything implementing
``start/exec/write_file/read_file/copy_in/copy_out/stop`` can back the
code builder. Phase 1 ships two implementations:

- :class:`~byok_agent.sandbox.docker.DockerSandbox` — real Docker
  containers (requires a Docker daemon).
- :class:`~byok_agent.sandbox.fake.FakeSandbox` — in-memory fake for
  tests and for environments without Docker.
"""
from .backend import (
    SandboxAbortedError,
    SandboxBackend,
    SandboxDeadError,
    SandboxError,
    SandboxResult,
    SandboxTimeoutError,
    run_with_abort,
)
from .docker import DockerSandbox, LANGUAGE_IMAGES, is_docker_available
from .fake import FakeSandbox

__all__ = [
    "SandboxAbortedError",
    "SandboxBackend",
    "SandboxDeadError",
    "SandboxError",
    "SandboxResult",
    "SandboxTimeoutError",
    "DockerSandbox",
    "FakeSandbox",
    "LANGUAGE_IMAGES",
    "is_docker_available",
    "run_with_abort",
]
