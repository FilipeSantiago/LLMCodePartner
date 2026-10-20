"""OpenSpec artifact IO on a registered, mounted project workspace.

The native counterpart of `BridgeMcpFileClient`/`DirectMcpFileClient`: the same three
`McpFileClient` methods, the same artifact-path boundary (`openspec/`, `.codepartner/`),
and `JetBrainsMcpError` for failures so `OpenSpecArtifactStore` handles both alike.
"""

from __future__ import annotations

import asyncio
import os
import tempfile

from agent.spec.jetbrains_mcp import JetBrainsMcpError

_ARTIFACT_ROOTS = ("openspec/", ".codepartner/")


class FilesystemFileClient:
    """Read and write OpenSpec artifacts directly in a project workspace."""

    def __init__(self, workspace: str):
        self._root = os.path.realpath(workspace)

    def _path(self, path: str) -> str:
        if not isinstance(path, str) or not path.startswith(_ARTIFACT_ROOTS):
            raise JetBrainsMcpError(f"OpenSpec artifact IO refuses non-artifact path {path!r}")
        full = os.path.realpath(os.path.join(self._root, path))
        # Realpath containment: a symlinked openspec/ cannot lead outside the project.
        relative = os.path.relpath(full, self._root) + "/"
        if (os.path.commonpath([full, self._root]) != self._root
                or not relative.startswith(_ARTIFACT_ROOTS)):
            raise JetBrainsMcpError(f"artifact path {path!r} resolves outside its project area")
        return full

    async def read_file(self, path: str) -> str:
        full = self._path(path)
        try:
            return await asyncio.to_thread(_read, full)
        except FileNotFoundError as exc:
            raise JetBrainsMcpError(f"File {path} doesn't exist or can't be opened") from exc
        except OSError as exc:
            raise JetBrainsMcpError(f"could not read {path}: {exc}") from exc

    async def file_exists(self, path: str) -> bool:
        return await asyncio.to_thread(os.path.isfile, self._path(path))

    async def create_file(self, path: str, content: str, overwrite: bool = True) -> None:
        full = self._path(path)
        try:
            await asyncio.to_thread(_write_atomic, full, content, overwrite)
        except FileExistsError as exc:
            raise JetBrainsMcpError(f"{path} already exists") from exc
        except OSError as exc:
            raise JetBrainsMcpError(f"could not write {path}: {exc}") from exc


def _read(full: str) -> str:
    with open(full, encoding="utf-8") as handle:
        return handle.read()


def _write_atomic(full: str, content: str, overwrite: bool) -> None:
    """Write via a temp file in the same directory and `os.replace`, so a reader (the
    IDE, a concurrent run) never sees a half-written catalog."""
    directory = os.path.dirname(full)
    os.makedirs(directory, exist_ok=True)
    if not overwrite and os.path.lexists(full):
        raise FileExistsError(full)
    mode = os.stat(full).st_mode & 0o777 if os.path.exists(full) else 0o644
    descriptor, temporary = tempfile.mkstemp(dir=directory, prefix=".codepartner-",
                                             suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, full)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
