"""Pin a test module to the IDE execution path.

Native execution is the default; modules that cover the IDE MCP path (no registered
project) call `pin()` from `setUpModule` and `unpin()` from `tearDownModule`.
"""

import os
from unittest import mock

_patch = mock.patch.dict(os.environ, {"CODEPARTNER_EXECUTION_MODE": "ide"})


def pin() -> None:
    _patch.start()


def unpin() -> None:
    _patch.stop()
