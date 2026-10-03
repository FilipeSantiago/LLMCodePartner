"""Temporary correlated DEBUG payload logging for the bridge investigation."""

import json
import logging
from typing import Any
from uuid import uuid4


def new_trace_id() -> str:
    # ##DELETE AFTER CORRECTION##
    return "trace_" + uuid4().hex


def debug(logger: logging.Logger, event: str, /, **fields: Any) -> None:
    """Emit one machine-readable diagnostic record without logging credentials.

    Callers deliberately pass request, tool, and source payloads for the temporary
    protocol diagnosis. HTTP headers and environment values are not accepted here.
    """
    # ##DELETE AFTER CORRECTION## Remove this full-payload diagnostic helper.
    logger.debug(
        "event=%s payload=%s",
        event,
        json.dumps(fields, ensure_ascii=False, default=repr, separators=(",", ":")),
    )
