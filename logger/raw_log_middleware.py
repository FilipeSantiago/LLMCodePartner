import os

from dotenv import load_dotenv

load_dotenv()


def _enabled() -> bool:
    return os.getenv("RAW_LOG_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")


class RawLogMiddleware:
    """Compatibility middleware.

    Request bodies, headers, prompts and tool arguments are never safe diagnostic
    material.  Keep this no-op so an old RAW_LOG_ENABLED setting cannot leak them.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        await self.app(scope, receive, send)
