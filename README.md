# vLLM-MLX (raullenchai fork)

**vllm-mlx fork optimized for Apple Silicon — tested with MiniMax-M2.5 and Qwen3-Coder-Next**

[![Fork](https://img.shields.io/badge/Fork-raullenchai%2Fvllm--mlx-orange?logo=github)](https://github.com/raullenchai/vllm-mlx)
[![Upstream](https://img.shields.io/badge/Upstream-waybarrios%2Fvllm--mlx-blue?logo=github)](https://github.com/waybarrios/vllm-mlx)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Apple Silicon](https://img.shields.io/badge/Apple-Silicon-black.svg)](https://support.apple.com/en-us/HT211814)

Built on [waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx) — GPU-accelerated LLM inference on Mac via [MLX](https://github.com/ml-explore/mlx). This fork adds 32 commits with MiniMax-M2.5 tool calling, reasoning separation, prompt caching, and more.

---

## What This Fork Adds

### MiniMax-M2.5 Specific

| Feature | Description |
|---------|-------------|
| MiniMax reasoning parser | Heuristic no-tag stripping for inline reasoning (0% leak rate, was 60%) |
| MiniMax tool call parser | Streaming + non-streaming XML tool call extraction |
| Auto-infer tool parser | `--reasoning-parser minimax` auto-selects the matching tool parser — zero extra flags |
| Chunk-boundary leak fix | Prevents tool call XML leaking into reasoning stream at chunk boundaries |
| Chinese reasoning patterns | Recognizes Chinese-language reasoning prefixes |
| Tool-use system prompt | Auto-injected instructions make model use tools proactively (100% call rate, was 67%) |

### General Improvements (all models)

| Feature | Description |
|---------|-------------|
| Prompt cache (SimpleEngine) | Persistent KV cache across requests — 10-15x faster multi-turn TTFT |
| Logprobs API | `logprobs` + `top_logprobs` per-token log probabilities |
| Streaming disconnect guard | Graceful handling of client disconnects mid-stream |
| Streaming reasoning buffer fix | Parser no longer eats content during buffer phase |
| Prompt cache EOS fix | Cache saved correctly on EOS for tool call responses |
| `developer` role normalization | `developer` → `system` for chat template compatibility |
| `prompt_tokens` reporting | Accurate token counts in usage response (was always 0) |
| Server crash prevention | Graceful fallback on malformed `response_format` schemas |
| `--prefill-step-size` flag | Configurable prefill chunk size for TTFT tuning |
| `--kv-bits` flag | KV cache quantization (4 or 8 bit) for long contexts |

### Original / Novel Work

| Feature | Description |
|---------|-------------|
| Tool logits bias | Jump-forward decoding for structured XML — 2-5x faster tool call generation |
| Structural tag constraints | JSON schema-aware parameter biasing for tool arguments |
| Frequency-aware cache eviction | LRU-LFU hybrid keeps system prompt blocks alive under pressure |
| Speculative decoding | `--draft-model` support for faster generation |
| Test suite | 300+ unit tests across parsers, engine, server, and tool calling |

---

## Quick Start

### 1. Install

```bash
pip install git+https://github.com/raullenchai/vllm-mlx.git
```

Or clone for development:

```bash
git clone https://github.com/raullenchai/vllm-mlx.git
cd vllm-mlx
pip install -e .
```

### 2. Pick a model and start the server

#### Qwen3-Coder-Next (recommended for coding agents)

Best balance of speed and intelligence. 32B dense model, excellent tool calling and code generation.

```bash
# Download (pick one quantization)
python -c "from mlx_lm import load; load('lmstudio-community/Qwen3-Coder-Next-MLX-6bit')"

# Start server
python3.12 -m vllm_mlx.server \
  --model lmstudio-community/Qwen3-Coder-Next-MLX-6bit \
  --tool-call-parser hermes \
  --prefill-step-size 8192 \
  --kv-bits 8 \
  --port 8000
```

Also available in 4bit (faster, slightly lower quality) and 8bit (slower, highest quality):

```bash
# 4bit — fastest, 42GB RAM, ~70 tok/s decode
--model lmstudio-community/Qwen3-Coder-Next-MLX-4bit

# 6bit — sweet spot, 60GB RAM, ~63 tok/s decode (recommended)
--model lmstudio-community/Qwen3-Coder-Next-MLX-6bit

# 8bit — highest quality, 75GB RAM, ~45 tok/s decode
--model lmstudio-community/Qwen3-Coder-Next-MLX-8bit
```

#### MiniMax-M2.5 (best reasoning quality)

229B MoE model with built-in reasoning. Best for complex multi-step reasoning tasks.

```bash
# Download
python -c "from mlx_lm import load; load('lmstudio-community/MiniMax-M2.5-MLX-4bit')"

# Start server
python3.12 -m vllm_mlx.server \
  --model lmstudio-community/MiniMax-M2.5-MLX-4bit \
  --reasoning-parser minimax \
  --prefill-step-size 4096 \
  --kv-bits 4 \
  --port 8000
```

`--reasoning-parser minimax` auto-enables the matching tool call parser and auto-tool-choice — zero extra flags needed.

> **Note:** MiniMax requires ~120GB RAM. Recommended for M3/M4 Ultra with 192GB+.

### 3. Test with curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is 15 * 37?"}]
  }'
```

### 4. Test with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is 15 * 37?"}],
)
print(response.choices[0].message.content)       # "555"
```

### 5. Tool calling

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        }
    }
}]

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
)

tool_call = response.choices[0].message.tool_calls[0]
print(tool_call.function.name)       # "get_weather"
print(tool_call.function.arguments)  # '{"city": "Tokyo"}'
```

---

## OpenClaw Integration

This fork was built to power [OpenClaw](https://github.com/openclaw) — an open-source coding agent.

Recommended `openclaw.json` model config:

```json
{
  "models": {
    "providers": {
      "vllm-mlx": {
        "baseUrl": "http://127.0.0.1:8000/v1",
        "apiKey": "no-key",
        "api": "openai-completions",
        "models": [{
          "id": "Qwen3-Coder-Next-MLX-6bit",
          "name": "Qwen3 Coder Next 6bit via vllm-mlx",
          "reasoning": false,
          "input": ["text"],
          "contextWindow": 40960,
          "maxTokens": 8192
        }]
      }
    }
  }
}
```

> For MiniMax-M2.5, set `"reasoning": true` to get reasoning traces in the response.

What works with OpenClaw:
- Streaming tool calling with reasoning separation
- Multi-turn prompt cache (22K+ tokens saved on cache hit, ~2s TTFT after first turn)
- Auto tool-use system prompt injection
- Accurate `prompt_tokens` for usage tracking

---

## Server Flags Reference

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | HuggingFace model name or local path | *(required)* |
| `--host` | Host to bind to | `127.0.0.1` |
| `--port` | Port to bind to | `8000` |
| `--reasoning-parser` | Reasoning parser: `minimax`, `qwen3`, `deepseek_r1`, `gpt_oss`, `harmony` | *(none)* |
| `--tool-call-parser` | Tool call parser: `hermes`, `minimax`, etc. Auto-enables `--enable-auto-tool-choice` | *(none)* |
| `--enable-auto-tool-choice` | Enable automatic tool choice (implied by `--tool-call-parser`) | off |
| `--enable-tool-logits-bias` | Jump-forward decoding bias for tool call structural tokens | off |
| `--continuous-batching` | Enable batched engine for multiple concurrent users | off |
| `--mllm` | Force loading as multimodal language model | auto-detect |
| `--mcp-config` | Path to MCP configuration file (JSON/YAML) | *(none)* |
| `--max-tokens` | Default max tokens for generation | model default |
| `--api-key` | API key for authentication | *(none — no auth)* |
| `--timeout` | Request timeout in seconds | `300` |
| `--rate-limit` | Requests per minute per client (0 = disabled) | `0` |
| `--embedding-model` | Pre-load an embedding model at startup | *(none)* |
| `--default-temperature` | Default temperature when not specified in request | model default |
| `--default-top-p` | Default top_p when not specified in request | model default |
| `--draft-model` | Draft model for speculative decoding (same tokenizer as main model) | *(none)* |
| `--num-draft-tokens` | Tokens to generate speculatively per step | `4` |
| `--prefill-step-size` | Tokens per prefill chunk (larger = faster TTFT if memory allows) | `2048` |
| `--kv-bits` | KV cache quantization: `4` or `8` bit | *(none — full precision)* |
| `--kv-group-size` | Group size for KV cache quantization | `64` |

**Example — Qwen3-Coder-Next:**

```bash
python3.12 -m vllm_mlx.server \
  --model lmstudio-community/Qwen3-Coder-Next-MLX-6bit \
  --tool-call-parser hermes \
  --prefill-step-size 8192 \
  --kv-bits 8 \
  --port 8000
```

**Example — MiniMax-M2.5:**

```bash
python3.12 -m vllm_mlx.server \
  --model lmstudio-community/MiniMax-M2.5-MLX-4bit \
  --reasoning-parser minimax \
  --prefill-step-size 4096 \
  --kv-bits 4 \
  --port 8000
```

---

## Performance

All benchmarks on **Mac Studio M3 Ultra (256GB)** with M3 Ultra's 800 GB/s memory bandwidth.

### Model Comparison

| Model | Quant | RAM | Decode | Prefill | Best For |
|-------|-------|-----|--------|---------|----------|
| Qwen3-Coder-Next | 4bit | 42GB | **70 tok/s** | 1270 tok/s | Speed-first, coding agents |
| Qwen3-Coder-Next | 6bit | 60GB | 63 tok/s | 1090-1440 tok/s | **Recommended** — best speed/quality balance |
| Qwen3-Coder-Next | 8bit | 75GB | ~45 tok/s | ~900 tok/s | Highest quality, still fast |
| MiniMax-M2.5 | 4bit | 120GB | 33-38 tok/s | 430-500 tok/s | Deep reasoning, complex tasks |

> **Why the speed difference?** Decode speed is memory-bandwidth-bound. M3 Ultra's 800 GB/s ÷ model size determines max throughput. Qwen3 6bit (60GB) ≈ 63 tok/s, MiniMax 4bit (120GB) ≈ 35 tok/s.

### Qwen3-Coder-Next-MLX-6bit (32B dense, 60GB)

| Metric | Value |
|--------|-------|
| Decode (non-streaming) | 63.2 tok/s |
| Decode (streaming) | 40-44 tok/s |
| Prefill | 1090-1440 tok/s |
| TTFT (cold, short prompt) | ~0.3s |
| TTFT (cache hit) | 0.3-0.5s |

### MiniMax-M2.5-MLX-4bit (229B MoE, 120GB)

| Metric | Value |
|--------|-------|
| Decode (128 tok) | 53 tok/s |
| Decode (512 tok) | 52 tok/s |
| Decode (2048 tok) | 50 tok/s |
| Decode (8192 tok) | 32 tok/s |
| TTFT (short ~50 tok) | 0.37s |
| TTFT (medium ~500 tok) | 0.79s |
| TTFT (long ~2K tok) | 1.42s |

### Prompt Cache (Multi-Turn)

SimpleEngine prompt cache (added in this fork) reuses KV cache across requests. On OpenClaw workloads with 22K+ token contexts: **23-30s TTFT → ~2s** (10-15x speedup).

| Turn | Without Cache | With Cache | Improvement |
|------|---------------|------------|-------------|
| Turn 1 (cold) | 0.52s | 0.52s | — |
| Turn 2 | 0.78s | 0.61s | **22% faster** |
| Turn 3 | 0.99s | 0.91s | **8% faster** |
| Turn 4 | 1.12s | 1.06s | **5% faster** |

### Tool Calling

| Test | Result | Latency |
|------|--------|---------|
| Single tool (weather) | Pass | 2.0s |
| Multi-arg (search) | Pass | 2.5s |
| Code execution | Pass | 4.0s |
| Multi-tool selection | Pass | 3.0s |

**4/4 accuracy** on both MiniMax and Qwen3-Coder-Next.

### Reasoning Quality (MiniMax Parser)

| Metric | Before (deepseek_r1) | After (minimax) |
|--------|---------------------|-----------------|
| Reasoning leak rate | 6/10 | **0/10** |
| Tool call success | 2/3 | **3/3** |
| prompt_tokens accuracy | 0/10 | **10/10** |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           vLLM API Layer                                │
│                    (OpenAI-compatible interface)                         │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            MLXPlatform                                  │
│               (vLLM platform plugin for Apple Silicon)                  │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
        ┌─────────────┬────────────┴────────────┬─────────────┐
        ▼             ▼                         ▼             ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│    mlx-lm     │ │   mlx-vlm     │ │   mlx-audio   │ │mlx-embeddings │
│(LLM inference)│ │ (Vision+LLM)  │ │  (TTS + STT)  │ │ (Embeddings)  │
└───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘
        │             │                         │             │
        └─────────────┴─────────────────────────┴─────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              MLX                                        │
│                (Apple ML Framework - Metal kernels)                      │
└─────────────────────────────────────────────────────────────────────────┘
```

**SimpleEngine** — Single-user mode. Calls mlx-lm directly with persistent prompt cache. Best for dedicated setups (e.g., one user + OpenClaw).

**BatchedEngine** — Multi-user mode (`--continuous-batching`). Uses Scheduler with paged KV cache and prefix sharing. Best for serving multiple concurrent clients.

---

## Upstream

This fork is based on [waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx). All upstream features (multimodal, audio, embeddings, Anthropic API, MCP) are available — see the [upstream README](https://github.com/waybarrios/vllm-mlx#readme) for full docs.

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
