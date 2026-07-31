import time

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class Model(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "dumb-router"


class ModelList(BaseModel):
    object: str = "list"
    data: list[Model]


@router.get("/v1/models", response_model=ModelList)
def list_models() -> ModelList:
    return ModelList(data=[Model(id="dumb-router", created=int(time.time()))])
