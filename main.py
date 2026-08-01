import asyncio
import logging

from dotenv import load_dotenv

# Load .env BEFORE importing modules that read env vars at import time (e.g.
# providers/ollama.py OLLAMA_MODEL/HOST, mcp_bridge/bridge.py MCP_TOOL_TIMEOUT).
load_dotenv()

from fastapi import FastAPI
from hypercorn.asyncio import serve
from hypercorn.config import Config

from api import dialog_controller, model_controller
from logger.raw_log_middleware import RawLogMiddleware

logging.basicConfig(level=logging.INFO)  # surface mcp_bridge tool logs

app = FastAPI(title="LLMCodePartner")

app.include_router(dialog_controller.router)
app.include_router(model_controller.router)

app.add_middleware(RawLogMiddleware)

if __name__ == "__main__":
    config = Config()
    config.bind = ["0.0.0.0:7777"]
    asyncio.run(serve(app, config))
