import uvicorn
from fastapi import FastAPI

from api import dialog_controller, model_controller

app = FastAPI(title="LLMCodePartner")

app.include_router(dialog_controller.router)
app.include_router(model_controller.router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7777)
