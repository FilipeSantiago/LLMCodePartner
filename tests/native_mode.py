"""Pin a test module to a stub native project.

Modules whose fakes stand in for the agents call `pin()` from `setUpModule` and
`unpin()` from `tearDownModule`: the planner and reviewer then see a bound, empty
native project instead of refusing to run without one.
"""

import tempfile
from pathlib import Path
from unittest import mock

from agent.workspace import NativePlan, NativeRun, RegisteredProject

_patches: list = []
_root: tempfile.TemporaryDirectory | None = None


def pin() -> None:
    global _root
    _root = tempfile.TemporaryDirectory()
    project = RegisteredProject("demo", _root.name, _root.name)
    _patches[:] = [
        mock.patch(f"{module}.plan_native", return_value=NativePlan(project))
        for module in ("agent.planning.service", "agent.routing.review")
    ] + [
        mock.patch(f"{module}.read_only_session",
                   side_effect=lambda _project: NativeRun(workspace=_root.name, read_only=True))
        for module in ("agent.planning.service", "agent.routing.review")
    ]
    for patch in _patches:
        patch.start()


def write(path: str, content: str) -> None:
    """Put a file into the stub project, where the reviewer reads it from disk."""
    target = Path(_root.name, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)


class native_pinned:
    """`pin()` for one `with` block."""

    def __enter__(self):
        pin()

    def __exit__(self, *_exc):
        unpin()


def unpin() -> None:
    for patch in reversed(_patches):
        patch.stop()
    _patches.clear()
    if _root is not None:
        _root.cleanup()
