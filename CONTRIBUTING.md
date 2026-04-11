# Contributing to Rapid-MLX

Thanks for your interest! Here's how to get started.

## Development Setup

```bash
# Clone and install in dev mode
git clone https://github.com/raullenchai/Rapid-MLX.git
cd Rapid-MLX
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest ruff        # dev tools for testing and linting

# Start a dev server
rapid-mlx serve qwen3.5-4b --port 8000
```

**Requirements:** Python 3.11+, macOS with Apple Silicon (M1/M2/M3/M4).

## Running Tests

```bash
# Run all unit tests (no model needed)
python3 -m pytest tests/ -x -q

# Run a specific test file
python3 -m pytest tests/test_tool_calling.py -v

# Lint and format
ruff check .
ruff format --check .
```

Most tests run without a model. Tests in `tests/test_event_loop.py` require a running server.

## Pull Request Workflow

1. Fork the repo and create a branch: `feat/`, `fix/`, `docs/`, `refactor/`
2. Make your changes with tests if applicable
3. Run `ruff check` and `ruff format` before committing
4. Open a PR against `main` with a clear description

## Ways to Contribute

### 🟢 Easy — No model download needed

- **Add a model alias** — Add a short name to `vllm_mlx/aliases.json` so users can `rapid-mlx serve <alias>` instead of typing a full HuggingFace path. See [open model-support issues](https://github.com/raullenchai/Rapid-MLX/issues?q=is%3Aissue+is%3Aopen+label%3Amodel-support).

- **Fix a `good first issue`** — Check the [good first issue](https://github.com/raullenchai/Rapid-MLX/labels/good%20first%20issue) label.

### 🟡 Medium — Needs a model + some testing

- **Test a model and report results** — Download a model, run benchmarks, report what works. Use the "Model Support Request" issue template.

- **Add parser auto-detection** — Add a regex pattern to `vllm_mlx/model_auto_config.py` so a new model family gets the right tool/reasoning parser automatically.

- **Verify client integrations** — Test Rapid-MLX with your favorite AI tool (Cursor, Continue, Aider, LangChain, etc.) and report results.

### 🔴 Advanced

- **Write a new tool call parser** — Add support for a new tool call format in `vllm_mlx/tool_parsers/`.
- **Performance optimization** — Profiling, kernel improvements, caching strategies.
- **BatchedEngine / continuous batching** — Multi-user serving improvements.

## How to Add a Model Alias

The easiest contribution — no model download needed!

**File:** `vllm_mlx/aliases.json`

```json
{
  "my-model-7b": "mlx-community/My-Model-7B-Instruct-4bit"
}
```

That's it. Find the MLX model on [HuggingFace mlx-community](https://huggingface.co/mlx-community) and add the mapping. Convention: `<family>-<size>` in lowercase (e.g., `qwen3.5-9b`, `gemma-4-26b`).

## How to Add Parser Auto-Detection

When users serve a model without `--tool-call-parser`, Rapid-MLX auto-detects the right parser from the model name.

**File:** `vllm_mlx/model_auto_config.py`

```python
# Add your pattern (order matters — more specific first):
(re.compile(r"my-model", re.IGNORECASE), ModelConfig(
    tool_call_parser="hermes",    # most common format
    reasoning_parser=None,        # set if model has thinking tags
)),
```

Common tool parsers: `hermes`, `llama`, `deepseek`, `gemma4`, `glm47`, `minimax`, `kimi`.
Common reasoning parsers: `qwen3`, `deepseek_r1`, `gemma4`, `minimax`.

**How to figure out the right parser:** Check the model's chat template for tool call format. Most models use Hermes-style `<tool_call>` tags. If unsure, try `hermes` first.

## Code Style

- We use `ruff` for linting and formatting
- Type hints are encouraged but not required
- Keep changes focused — one feature/fix per PR
