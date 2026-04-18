<p align="center">
  <img src="https://raw.githubusercontent.com/raullenchai/Rapid-MLX/main/docs/assets/logo.png" alt="Rapid-MLX" width="200">
</p>

<h1 align="center">Rapid-MLX</h1>

<p align="center">
  <strong>The fastest way to run AI on your Mac.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://pypi.org/project/rapid-mlx/"><img src="https://img.shields.io/pypi/v/rapid-mlx.svg" alt="PyPI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="tests/"><img src="https://img.shields.io/badge/tests-2100%2B-brightgreen.svg" alt="Tests"></a>
  <a href="https://support.apple.com/en-us/HT211814"><img src="https://img.shields.io/badge/Apple_Silicon-M1%20|%20M2%20|%20M3%20|%20M4-black.svg?logo=apple" alt="Apple Silicon"></a>
</p>

<p align="center">
  OpenAI-compatible inference server for Apple Silicon.<br>
  2-3x faster than Ollama. Works with Cursor, Claude Code, and any OpenAI app.
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#works-with">Integrations</a> &middot;
  <a href="#benchmarks">Benchmarks</a> &middot;
  <a href="#choose-your-model">Models</a>
</p>

---

### Why Rapid-MLX?

Ollama and llama.cpp are built for CUDA GPUs. On Apple Silicon, they copy data between CPU and GPU memory. Rapid-MLX uses Apple's [MLX framework](https://github.com/ml-explore/mlx) with native Metal kernels — **zero copies, 2-3x faster**.

| Problem | Rapid-MLX |
|---------|-----------|
| Ollama is slow on Mac | 2-3x faster on same models |
| Tool calling breaks at 4-bit | Auto-recovery across 17 parser formats |
| No prompt cache on hybrid models | DeltaNet state snapshots (2-4x TTFT) |
| One model at a time | Multi-model serving with request routing |

<p align="center">
  <img src="https://raw.githubusercontent.com/raullenchai/Rapid-MLX/main/docs/assets/demo.gif" alt="Rapid-MLX demo — install, serve Gemma 4, chat, tool calling" width="700">
</p>

| Your Mac | Model | Speed | What works |
|:---|:---|---:|:---|
| **16 GB** MacBook Air | Qwen3.5-4B | 168 tok/s | Chat, coding, tools |
| **32 GB** Mac Mini / Studio | Nemotron-Nano 30B | 141 tok/s | Fastest 30B, 100% tools |
| **32 GB** Mac Mini / Studio | Qwen3.6-35B | 95 tok/s | 256 experts, 262K context |
| **64 GB** Mac Mini / Studio | Qwen3.5-35B | 83 tok/s | Best balance of smart + fast |
| **96+ GB** Mac Studio / Pro | Qwen3.5-122B | 57 tok/s | Frontier-level intelligence |

---

## Quick Start

```bash
# Install
brew install raullenchai/rapid-mlx/rapid-mlx    # or: pip install rapid-mlx

# Serve
rapid-mlx serve qwen3.5-9b

# Chat (in another terminal)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Say hello"}]}'
```

That's it — OpenAI-compatible server on `localhost:8000`. Point any app at `http://localhost:8000/v1`.

<details>
<summary><b>More install options</b></summary>

```bash
pip install rapid-mlx                             # pip
curl -fsSL https://raullenchai.github.io/Rapid-MLX/install.sh | bash  # auto-setup
git clone https://github.com/raullenchai/Rapid-MLX.git && pip install -e .  # dev
pip install 'rapid-mlx[vision]'                   # + vision models
pip install 'rapid-mlx[audio]'                    # + TTS/STT
```

> **"No matching distribution" error?** Python too old. `brew install python@3.12` then `python3.12 -m pip install rapid-mlx`

</details>

**Python SDK:**

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Say hello"}],
)
print(response.choices[0].message.content)
```

---

## Works With

### Agents & Frameworks

| | Name | Setup |
|:---|:---|:---|
| **Cursor** | IDE | Settings → OpenAI Base URL → `http://localhost:8000/v1` |
| **Claude Code** | Agent | `OPENAI_BASE_URL=http://localhost:8000/v1 claude` |
| **Aider** | Agent | `aider --openai-api-base http://localhost:8000/v1` |
| **Hermes** | Agent (64K stars) | [config](tests/integrations/test_hermes.py) — 62 tools, multi-turn |
| **PydanticAI** | Framework | [example](tests/integrations/test_pydantic_ai_full.py) — typed agents, structured output |
| **LangChain** | Framework | [example](tests/integrations/test_langchain.py) — `ChatOpenAI`, tools, streaming |
| **smolagents** | Framework (HF) | [example](tests/integrations/test_smolagents_full.py) — CodeAgent + ToolCallingAgent |
| **Goose** | Agent | `GOOSE_PROVIDER=ollama OLLAMA_HOST=http://localhost:8000 goose` |
| **Open WebUI** | UI | Docker → `OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1` |
| **Continue.dev** | IDE | VS Code / JetBrains extension |
| **Any OpenAI app** | — | Point at `http://localhost:8000/v1` |

> Run `rapid-mlx agents` to see all 11 supported agents with setup guides.

<details>
<summary><b>Full setup examples</b></summary>

**Cursor:** Settings → Models → Add Model:
```
OpenAI API Base:  http://localhost:8000/v1
API Key:          not-needed
Model name:       default
```

**Hermes Agent** (`~/.hermes/config.yaml`):
```yaml
model:
  provider: "custom"
  default: "default"
  base_url: "http://localhost:8000/v1"
  context_length: 32768
```

**PydanticAI:**
```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIChatModel(
    model_name="default",
    provider=OpenAIProvider(base_url="http://localhost:8000/v1", api_key="not-needed"),
)
agent = Agent(model)
print(agent.run_sync("What is 2+2?").output)
```

**smolagents:**
```python
from smolagents import CodeAgent, OpenAIServerModel

model = OpenAIServerModel(model_id="default", api_base="http://localhost:8000/v1", api_key="not-needed")
agent = CodeAgent(tools=[], model=model)
agent.run("What is 5 multiplied by 7?")
```

**Open WebUI:**
```bash
docker run -d -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -e ENABLE_OLLAMA_API=False \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY=not-needed \
  -v open-webui:/app/backend/data \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main
```

**Anthropic SDK:**
```python
from anthropic import Anthropic
client = Anthropic(base_url="http://localhost:8000", api_key="not-needed")
message = client.messages.create(model="default", max_tokens=1024, messages=[{"role": "user", "content": "Say hello"}])
print(message.content[0].text)
```

</details>

### Model-Harness Index (MHI)

Which model works best with which agent framework? MHI = 50% tool calling + 30% HumanEval + 20% MMLU.

| Model | Hermes | PydanticAI | LangChain | smolagents | Speed |
|---|:---:|:---:|:---:|:---:|---:|
| **Qwopus 27B** | 92 | 92 | 92 | 92 | 38 tok/s |
| **Qwen3.5 27B** | 82 | 82 | 82 | — | 38 tok/s |
| **Nemotron Nano 30B** | 58 | 59 | 59 | 58 | 141 tok/s |
| **Gemma 4 26B** | 62 | 45 | — | 62 | 85 tok/s |
| **DeepSeek-R1 32B** | 57 | 54 | — | 79 | ~30 tok/s |
| **Llama 3.3 70B** | 56 | 67 | 67 | 83 | ~20 tok/s |

> **Best overall:** Qwopus 27B (MHI 92). **Fastest:** Nemotron Nano 30B (141 tok/s). **Best for smolagents:** Llama 3.3 70B (83). Run `rapid-mlx agents <name> --test` to benchmark on your setup.

---

## Benchmarks

Tested on **Mac Studio M3 Ultra (256GB)**. Ollama v0.20.4 (MLX backend).

| Model | Rapid-MLX | Ollama / Alternative | Speedup |
|-------|----------|---------------------|---------|
| Phi-4 Mini 14B | **180** tok/s | 56 (Ollama) | **3.2x** |
| Qwen3.5-4B | **168** tok/s | 155 (mlx-lm) | 1.1x |
| Nemotron-Nano 30B | **141** tok/s | — | — |
| GPT-OSS 20B | **127** tok/s | 79 (mlx-lm) | **1.6x** |
| Qwen3.5-9B | **108** tok/s | 41 (Ollama) | **2.6x** |
| Qwen3.6-35B | **95** tok/s | — | — |
| Gemma 4 26B | **85** tok/s | 68 (Ollama) | **1.3x** |
| Qwen3.5-35B | **83** tok/s | 75 (oMLX) | 1.1x |
| Qwen3.5-122B | **44** tok/s | 43 (mlx-lm) | ~1.0x |

<details>
<summary><b>TTFT, capability comparison, eval benchmarks</b></summary>

**Prompt Cache — TTFT Advantage:**

| Model | Cold TTFT | Cached TTFT | Speedup |
|-------|-----------|-------------|---------|
| Qwen3-Coder 80B | 0.66s | **0.16s** | **4.3x** |
| Qwen3.5-35B | 0.49s | **0.19s** | **2.6x** |
| Qwen3.5-27B | 0.58s | **0.27s** | **2.1x** |

**Capability Comparison:**

| Feature | Rapid-MLX | Ollama | llama.cpp | mlx-lm |
|---------|:---------:|:------:|:---------:|:------:|
| Tool calling (100%) | Yes | Yes | Partial | No |
| Auto-recovery | Yes | Yes | Yes | No |
| Prompt cache (KV + DeltaNet) | Yes | No | No | No |
| Vision + Audio | Yes | Yes | No | No |
| 17 tool parsers | Yes | No | No | No |
| Streaming | Yes | Yes | Yes | Yes |

**Eval benchmarks (20 models):**

| Model | Speed | Tools | Code | Reasoning | General | Avg |
|-------|-------|-------|------|-----------|---------|-----|
| Qwen3.5-122B 8bit | 44 t/s | 87% | 90% | 90% | 90% | **89%** |
| Qwen3.5-35B 8bit | 83 t/s | 90% | 90% | 80% | 80% | 85% |
| Qwen3.5-9B | 108 t/s | 83% | 70% | 60% | 70% | 71% |
| GPT-OSS 20B | 127 t/s | 80% | 20% | 60% | 90% | 62% |
| Qwen3.5-4B | 168 t/s | 73% | 50% | 50% | 50% | 56% |

</details>

---

## Choose Your Model

| Your Mac | Model | RAM | Speed | Quality |
|----------|-------|-----|-------|---------|
| **16 GB** | `rapid-mlx serve qwen3.5-4b` | 2.4 GB | 168 tok/s | Good for chat |
| **24 GB** | `rapid-mlx serve qwen3.5-9b` | 5.1 GB | 108 tok/s | Great all-rounder |
| **32 GB** | `rapid-mlx serve nemotron-30b` | 18 GB | 141 tok/s | Fastest 30B |
| **32 GB** | `rapid-mlx serve qwen3.6-35b` | 20 GB | 95 tok/s | 262K context |
| **64 GB** | `rapid-mlx serve qwen3.5-35b` | 37 GB | 83 tok/s | **Sweet spot** |
| **96+ GB** | `rapid-mlx serve qwen3.5-122b` | 65 GB | 57 tok/s | Frontier-level |

> Run `rapid-mlx models` to see all available aliases. Vision: `rapid-mlx serve qwen3-vl-4b --mllm`

<details>
<summary><b>Parser auto-detection</b></summary>

Parsers are auto-detected from the model name — no flags needed for supported families.

| Model Family | Tool Parser | Reasoning Parser |
|-------------|-------------|-----------------|
| Qwen3.5 | `hermes` | `qwen3` |
| Qwen3.6 | `qwen3_coder_xml` | `qwen3` |
| DeepSeek R1 | `deepseek` / `deepseek_v31` | `deepseek_r1` |
| GLM-4.7 | `glm47` | — |
| Gemma 4 | `hermes` | — |
| Llama 3.x | `llama` | — |
| Mistral | `hermes` | — |

All 17 parsers include automatic recovery for broken tool calls.

</details>

---

## Features

- **Tool Calling** — 17 parser formats, auto-recovery, streaming, tool logits bias
- **Reasoning** — Separate `reasoning_content` field for Qwen3, DeepSeek-R1, MiniMax
- **Prompt Cache** — KV trimming + DeltaNet state snapshots, 2-5x faster TTFT
- **Multimodal** — Vision, audio (STT/TTS), embeddings via same API
- **Cloud Routing** — Auto-route large requests to GPT-5/Claude when local is slow
- **Continuous Batching** — Serve multiple users concurrently (default)

<details>
<summary><b>Server flags reference</b></summary>

| Flag | Description | Default |
|------|-------------|---------|
| `--simple-engine` | Legacy single-user mode | off |
| `--tool-call-parser` | Override auto-detected parser | auto |
| `--reasoning-parser` | Override reasoning parser | auto |
| `--enable-tool-logits-bias` | Faster structured output | off |
| `--prefill-step-size` | Tokens per prefill chunk | 2048 |
| `--kv-bits` | KV cache quantization (4/8) | full |
| `--enable-prefix-cache` | Cross-request prefix cache | off |
| `--gpu-memory-utilization` | Metal memory fraction | 0.90 |
| `--cloud-model` | Cloud LLM for routing | disabled |
| `--api-key` | API authentication | no auth |
| `--rate-limit` | Requests/min per client | unlimited |
| `--mcp-config` | MCP tool config file | none |

</details>

<details>
<summary><b>Troubleshooting</b></summary>

- **Out of memory** — Model too big. Use `--kv-bits 4` or pick a smaller model.
- **Empty responses** — Thinking models use tokens on reasoning. Try `--no-thinking`.
- **Tool calls as plain text** — Auto-recovery handles most cases. Set `--tool-call-parser` for edge cases.
- **Slow first response** — Add `--prefill-step-size 8192`. Subsequent turns hit prompt cache.

</details>

---

## Roadmap

| Technique | Expected Gain | Status |
|-----------|---------------|--------|
| [Standard Speculative Decode](https://arxiv.org/abs/2302.01318) — draft model acceleration | 1.5-2.3x decode | Not started |
| [EAGLE-3](https://arxiv.org/abs/2503.01840) — feature-level draft on Metal | 3-6.5x decode | Not started |
| [ReDrafter](https://arxiv.org/abs/2403.09919) — Apple's RNN draft head | 1.4-1.5x decode | Not started |

---

## Contributing

```bash
git clone https://github.com/raullenchai/Rapid-MLX.git
cd Rapid-MLX && pip install -e '.[dev]'
make test       # run 2100+ tests
make smoke      # doctor smoke tier
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. Easy first issues: [add a model alias](https://github.com/raullenchai/Rapid-MLX/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22), [report a bug](https://github.com/raullenchai/Rapid-MLX/issues/new?template=bug_report.yml).

<a href="https://github.com/raullenchai/Rapid-MLX/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=raullenchai/Rapid-MLX" />
</a>

## License

Apache 2.0 — see [LICENSE](LICENSE).
