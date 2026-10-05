import time

from fastapi import APIRouter
from pydantic import BaseModel

from agent.spec.modes import MODES

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
    # Each mode is a virtual model so the IDE chat model picker selects it.
    created = int(time.time())
    return ModelList(data=[Model(id=mode.model_id, created=created) for mode in MODES])
