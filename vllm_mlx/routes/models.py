# SPDX-License-Identifier: Apache-2.0
"""Model listing endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from ..api.models import ModelInfo, ModelsResponse
from ..server import verify_api_key

router = APIRouter()


@router.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models() -> ModelsResponse:
    """List available models (supports multi-model)."""
    from ..server import _model_alias, _model_name, _model_registry

    models = []
    if _model_registry:
        for entry in _model_registry.list_entries():
            models.append(ModelInfo(id=entry.model_name))
            for alias in sorted(entry.aliases):
                if alias != entry.model_name:
                    models.append(ModelInfo(id=alias))
    elif _model_name:
        models.append(ModelInfo(id=_model_name))
        if _model_alias and _model_alias != _model_name:
            models.append(ModelInfo(id=_model_alias))
    return ModelsResponse(data=models)


@router.get("/v1/models/{model_id}", dependencies=[Depends(verify_api_key)])
async def retrieve_model(model_id: str) -> ModelInfo:
    """Retrieve a specific model by ID."""
    from ..server import _model_alias, _model_name, _model_registry

    if _model_registry and model_id in _model_registry:
        return ModelInfo(id=model_id)
    if model_id in (_model_name, _model_alias):
        return ModelInfo(id=model_id)
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
