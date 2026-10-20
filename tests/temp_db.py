"""Point the default routing database (and the IDE session store) at a temp file.

Modules whose code paths persist IDE session identity call `start()` from
`setUpModule` and `stop()` from `tearDownModule`, so tests never write to the
user's `~/.codepartner/routing.sqlite3`.
"""

import os
import tempfile
from pathlib import Path
from unittest import mock

_state: dict = {}


def start() -> None:
    directory = tempfile.TemporaryDirectory()
    patch = mock.patch.dict(os.environ, {
        "ROUTING_DB_PATH": str(Path(directory.name) / "routing.sqlite3")})
    patch.start()
    _state.update(directory=directory, patch=patch)


def stop() -> None:
    _state.pop("patch").stop()
    _state.pop("directory").cleanup()
