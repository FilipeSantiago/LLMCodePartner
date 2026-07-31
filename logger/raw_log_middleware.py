import os

from dotenv import load_dotenv

load_dotenv()


def _enabled() -> bool:
    return os.getenv("RAW_LOG_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")


class RawLogMiddleware:
    """ASGI middleware that prints the raw request scope + body.

    Toggled by the RAW_LOG_ENABLED env var (.env). When disabled it is a
    transparent pass-through.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if not _enabled() or scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)

        print("RAW scope:", scope["type"], scope.get("method"), scope["path"], scope["headers"])

        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        messages, body, more = [], b"", True
        while more:
            message = await receive()
            messages.append(message)
            if message["type"] == "http.request":
                body += message.get("body", b"")
                more = message.get("more_body", False)
            else:
                more = False
        print("RAW body:", body)

        it = iter(messages)

        async def replay():
            try:
                return next(it)
            except StopIteration:
                return await receive()

        await self.app(scope, replay, send)
