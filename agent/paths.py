"""Recognizing file paths in free text, shared by discovery, contracts and subagent metrics."""

from __future__ import annotations

import re

_EXTENSIONS = ("py|pyi|ipynb|js|jsx|mjs|ts|tsx|java|kt|kts|go|rs|rb|c|h|cc|cpp|hpp|cs|swift|"
               "scala|sql|sh|md|rst|txt|yaml|yml|json|toml|cfg|ini|xml|html|css|gradle|properties")
# A known source or config extension, or a dotfile (`.env.example`), so abbreviations
# ("e.g") and dotted symbols ("Model.predict") are not paths.
_PATH = re.compile(rf"(?<![\w.-])(/?(?:[\w.-]+/)*(?:[\w-][\w.-]*\.(?:{_EXTENSIONS})"
                   r"|\.[\w-]+(?:\.[\w-]+)*\.[A-Za-z0-9]+))(?![\w/-])")


def norm(path: str) -> str:
    path = path.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path


def paths_in(text: str) -> list[str]:
    """Path-like tokens with a source or config extension."""
    return list(dict.fromkeys(norm(match.group(1)) for match in _PATH.finditer(text or "")))
