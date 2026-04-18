# SPDX-License-Identifier: Apache-2.0
"""Model listing endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from ..api.models import ModelInfo, ModelsResponse
from ..config import get_config
from ..middleware.auth import verify_api_key

router = APIRouter()


@router.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models() -> ModelsResponse:
    """List available models (supports multi-model)."""
    cfg = get_config()

    models = []
    if cfg.model_registry:
        for entry in cfg.model_registry.list_entries():
            models.append(ModelInfo(id=entry.model_name))
            for alias in sorted(entry.aliases):
                if alias != entry.model_name:
                    models.append(ModelInfo(id=alias))
    elif cfg.model_name:
        models.append(ModelInfo(id=cfg.model_name))
        if cfg.model_alias and cfg.model_alias != cfg.model_name:
            models.append(ModelInfo(id=cfg.model_alias))
    return ModelsResponse(data=models)


@router.get("/v1/models/{model_id}", dependencies=[Depends(verify_api_key)])
async def retrieve_model(model_id: str) -> ModelInfo:
    """Retrieve a specific model by ID."""
    cfg = get_config()

    if cfg.model_registry and model_id in cfg.model_registry:
        return ModelInfo(id=model_id)
    if model_id in (cfg.model_name, cfg.model_alias):
        return ModelInfo(id=model_id)
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
