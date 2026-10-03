import asyncio

from dotenv import load_dotenv

# Load .env BEFORE importing modules that read env vars at import time (e.g.
# providers/ollama.py OLLAMA_MODEL/HOST, mcp_bridge/bridge.py MCP_TOOL_TIMEOUT).
load_dotenv()

from fastapi import FastAPI
from hypercorn.asyncio import serve
from hypercorn.config import Config

from api import dialog_controller, model_controller
from logger.configuration import configure_logging
from mcp_bridge import mcp_http

configure_logging()

app = FastAPI(title="LLMCodePartner")

app.include_router(dialog_controller.router)
app.include_router(model_controller.router)

# Bridged tools for out-of-process providers (codex): one private, short-lived
# `/mcp/<token>` endpoint per Run — see mcp_bridge/mcp_http.py.
app.mount("/mcp", mcp_http.asgi_app)

if __name__ == "__main__":
    config = Config()
    config.bind = ["0.0.0.0:7777"]
    asyncio.run(serve(app, config))
