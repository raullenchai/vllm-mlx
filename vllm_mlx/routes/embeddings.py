# SPDX-License-Identifier: Apache-2.0
"""Embeddings endpoint."""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException

from ..api.models import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)
from ..server import check_rate_limit, verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/v1/embeddings",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """Create embeddings for the given input text(s)."""
    from ..server import (
        _embedding_engine,
        _embedding_model_locked,
        load_embedding_model,
    )

    try:
        model_name = request.model

        if (
            _embedding_model_locked is not None
            and model_name != _embedding_model_locked
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Embedding model '{model_name}' is not available. "
                    f"This server was started with --embedding-model {_embedding_model_locked}. "
                    f"Only '{_embedding_model_locked}' can be used for embeddings. "
                    f"Restart the server with a different --embedding-model to use '{model_name}'."
                ),
            )

        load_embedding_model(model_name, lock=False, reuse_existing=True)

        texts = request.input if isinstance(request.input, list) else [request.input]

        if not texts:
            raise HTTPException(status_code=400, detail="Input must not be empty")

        start_time = time.perf_counter()
        prompt_tokens = _embedding_engine.count_tokens(texts)
        embeddings = _embedding_engine.embed(texts)
        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Embeddings: {len(texts)} inputs, {prompt_tokens} tokens in {elapsed:.2f}s"
        )

        data = [
            EmbeddingData(index=i, embedding=vec) for i, vec in enumerate(embeddings)
        ]

        return EmbeddingResponse(
            data=data,
            model=model_name,
            usage=EmbeddingUsage(
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens,
            ),
        )

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="mlx-embeddings not installed. Install with: pip install 'rapid-mlx[embeddings]'",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
