# SPDX-License-Identifier: Apache-2.0
"""
Embedding engine using mlx-embeddings.

Provides lazy-loaded model management and batch embedding generation
for the OpenAI-compatible /v1/embeddings endpoint.
"""

import logging
import time

import mlx.core as mx

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Wrapper around mlx-embeddings for text embedding generation.

    Supports lazy model loading and batch embedding with proper
    tokenization and pooling.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the embedding model and tokenizer."""
        from mlx_embeddings import load

        logger.info(f"Loading embedding model: {self.model_name}")
        start = time.perf_counter()
        self._model, self._tokenizer = load(self.model_name)
        elapsed = time.perf_counter() - start
        logger.info(f"Embedding model loaded in {elapsed:.2f}s: {self.model_name}")

    def _ensure_loaded(self) -> None:
        if not self.is_loaded:
            self.load()

    def embed(self, texts: str | list[str]) -> list[list[float]]:
        """
        Generate embeddings for one or more texts.

        Args:
            texts: A single string or list of strings.

        Returns:
            List of embedding vectors (one per input text).
        """
        self._ensure_loaded()

        if isinstance(texts, str):
            texts = [texts]

        # Tokenize directly instead of using mlx_embeddings.generate(),
        # which has compatibility issues with newer tokenizers (e.g.
        # GemmaTokenizer lacks batch_encode_plus, and the model's __call__
        # expects positional `inputs` not `input_ids` as a kwarg).
        inner_tok = getattr(self._tokenizer, "_tokenizer", self._tokenizer)
        encoded = inner_tok(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )

        input_ids = mx.array(encoded["input_ids"])
        attention_mask = mx.array(encoded["attention_mask"])

        output = self._model(input_ids, attention_mask=attention_mask)

        # text_embeds shape: (batch_size, embedding_dim)
        embeds: mx.array = output.text_embeds

        # Convert to Python lists for JSON serialization
        return embeds.tolist()

    def count_tokens(self, texts: str | list[str]) -> int:
        """Approximate token count for usage reporting."""
        self._ensure_loaded()

        if isinstance(texts, str):
            texts = [texts]

        total = 0
        for text in texts:
            try:
                tokens = self._tokenizer.encode(text)
                if isinstance(tokens, list) or hasattr(tokens, "__len__"):
                    total += len(tokens)
                else:
                    total += tokens.size
            except Exception:
                # Fallback: rough estimate of ~4 chars per token
                total += max(1, len(text) // 4)
        return total
