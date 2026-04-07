# Integration Tests

End-to-end tests that exercise Rapid-MLX from a real client library.

These are **not** run as part of `pytest tests/` because they need:

1. A running Rapid-MLX server on `http://localhost:8000`
2. A loaded model (Gemma 4 26B by default — see each test's `MODEL_ID`)
3. The client library installed (varies per test)
4. For `test_librechat_docker.py`: Docker + a temporary LibreChat stack

## Running

Start the server first:

```bash
rapid-mlx serve "/path/to/your/mlx-model" \
    --tool-call-parser gemma4 --enable-auto-tool-choice
```

Then run any test:

```bash
# Python integrations (need pip install of the client library)
python3 tests/integrations/test_pydantic_ai_full.py
python3 tests/integrations/test_smolagents_full.py
python3 tests/integrations/test_langchain.py
python3 tests/integrations/test_anthropic_sdk.py

# CLI integrations
bash tests/integrations/test_aider.sh

# Docker E2E (LibreChat)
python3 tests/integrations/test_librechat_docker.py
```

`test_librechat_docker.py` assumes a `docker-compose.yml` and `librechat.yaml`
on `http://localhost:3081` — see the test docstring for the setup it expects.

## Coverage matrix

| Test | Plain | Stream | Tool call | Multi-tool | Structured | Notes |
|---|---|---|---|---|---|---|
| `test_pydantic_ai_full.py` | ✓ | ✓ | ✓ | ✓ | ✓ | + multi-turn |
| `test_smolagents_full.py` | ✓ | — | ✓ | ✓ | — | CodeAgent + ToolCallingAgent |
| `test_langchain.py` | ✓ | ✓ | ✓ | ✓ | ✓ | + system prompt |
| `test_anthropic_sdk.py` | ✓ | ✓ | ✓ | — | — | `/v1/messages` endpoint |
| `test_aider.sh` | — | — | — | — | — | Edit-and-write workflow |
| `test_librechat_docker.py` | — | — | — | — | — | Container, auth, model fetch |

The model used by these tests is hard-coded to a local path. Edit the
`MODEL_ID` constant in each script if you're running a different model.
