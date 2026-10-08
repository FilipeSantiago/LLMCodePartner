"""Cheap validation of verified IDE postimages; never modifies project files."""

import ast
import json
from collections.abc import Callable


class ValidatedResult(str):
    """Keep the IDE's return value while carrying validation to the agent layer."""

    def __new__(cls, value: str, validation: str):
        instance = super().__new__(cls, value)
        instance.validation = validation
        return instance


VALIDATORS: dict[str, Callable[[str, str], None]] = {
    ".py": lambda content, path: ast.parse(content, filename=path),
    ".json": lambda content, _path: json.loads(content),
}


def validator_for(path: str) -> Callable[[str, str], None] | None:
    return next((validator for suffix, validator in VALIDATORS.items()
                 if path.endswith(suffix)), None)


def validate_postimages(postimages: dict[str, str | None]) -> str:
    """Return an explicit result for each edited source file.

    This only checks syntax and parsability. Import resolution, lint and tests are
    separate checks and must not be claimed by this function.
    """
    messages = []
    for path, content in postimages.items():
        if content is None:
            messages.append(f"{path}: no postimage content; parser check skipped")
            continue
        try:
            validator = validator_for(path)
            if validator is None:
                messages.append(f"{path}: no configured parser; check skipped")
            else:
                validator(content, path)
                label = "Python" if path.endswith(".py") else "JSON" if path.endswith(".json") else "file"
                messages.append(f"{path}: {label} syntax valid")
        except (SyntaxError, json.JSONDecodeError) as exc:
            line = getattr(exc, "lineno", None)
            reason = getattr(exc, "msg", str(exc))
            context = content.splitlines()
            excerpt = ""
            if isinstance(line, int) and 1 <= line <= len(context):
                excerpt = f"; {line} | {context[line - 1][:160]}"
            messages.append(f"{path}: VALIDATION FAILED at line {line}: {reason}{excerpt}")
    return "Validation:\n" + ("\n".join(messages) if messages else "No postimages available; check skipped")
