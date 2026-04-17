# Embeddings

rapid-mlx supports text embeddings using [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings), providing an OpenAI-compatible `/v1/embeddings` endpoint.

## Installation

```bash
pip install mlx-embeddings>=0.0.5
```

## Quick Start

### Start the server with an embedding model

```bash
# Pre-load a specific embedding model at startup
rapid-mlx serve my-llm-model --embedding-model mlx-community/all-MiniLM-L6-v2-4bit
```

If you don't use `--embedding-model`, the embedding model is loaded lazily on the first request.

### Generate embeddings with the OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Single text
response = client.embeddings.create(
    model="mlx-community/all-MiniLM-L6-v2-4bit",
    input="Hello world"
)
print(response.data[0].embedding[:5])  # First 5 dimensions

# Batch of texts
response = client.embeddings.create(
    model="mlx-community/all-MiniLM-L6-v2-4bit",
    input=[
        "I love machine learning",
        "Deep learning is fascinating",
        "Natural language processing rocks"
    ]
)
for item in response.data:
    print(f"Text {item.index}: {len(item.embedding)} dimensions")
```

### Using curl

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/all-MiniLM-L6-v2-4bit",
    "input": ["Hello world", "How are you?"]
  }'
```

## Supported Models

Any BERT, XLM-RoBERTa, or ModernBERT model from HuggingFace that is compatible with mlx-embeddings:

| Model | Use Case | Size |
|-------|----------|------|
| `mlx-community/all-MiniLM-L6-v2-4bit` | Fast, compact | Small |
| `mlx-community/embeddinggemma-300m-6bit` | High quality | 300M |
| `mlx-community/bge-large-en-v1.5-4bit` | Best for English | Large |

## Model Management

### Lazy loading

By default, the embedding model is loaded on the first `/v1/embeddings` request. You can switch models between requests and the previous model will be unloaded automatically.

### Pre-loading at startup

Use `--embedding-model` to load a model at startup. When this flag is set, only that specific model can be used for embeddings:

```bash
rapid-mlx serve my-llm-model --embedding-model mlx-community/all-MiniLM-L6-v2-4bit
```

Requesting a different model will return a 400 error.

## API Reference

### POST /v1/embeddings

Create embeddings for the given input text(s).

**Request body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Model name from HuggingFace |
| `input` | string or list[string] | Yes | Text(s) to embed |

**Response:**

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.023, -0.982, ...]},
    {"object": "embedding", "index": 1, "embedding": [0.112, -0.543, ...]}
  ],
  "model": "mlx-community/all-MiniLM-L6-v2-4bit",
  "usage": {"prompt_tokens": 12, "total_tokens": 12}
}
```

## Python API

### Direct usage without server

```python
from vllm_mlx.embedding import EmbeddingEngine

engine = EmbeddingEngine("mlx-community/all-MiniLM-L6-v2-4bit")
engine.load()

vectors = engine.embed(["Hello world", "How are you?"])
print(f"Dimensions: {len(vectors[0])}")

tokens = engine.count_tokens(["Hello world"])
print(f"Token count: {tokens}")
```

## Troubleshooting

### mlx-embeddings not installed

```
pip install mlx-embeddings>=0.0.5
```

### Model not found

Make sure the model name matches a HuggingFace repository compatible with mlx-embeddings. You can pre-download models:

```bash
huggingface-cli download mlx-community/all-MiniLM-L6-v2-4bit
```
