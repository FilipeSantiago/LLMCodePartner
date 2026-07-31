import asyncio

from fastapi import FastAPI
from hypercorn.asyncio import serve
from hypercorn.config import Config

from api import dialog_controller, model_controller

app = FastAPI(title="LLMCodePartner")

app.include_router(dialog_controller.router)
app.include_router(model_controller.router)

if __name__ == "__main__":
    config = Config()
    config.bind = ["0.0.0.0:7777"]
    asyncio.run(serve(app, config))
