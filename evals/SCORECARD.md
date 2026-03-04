# vllm-mlx Model Scorecard

*Auto-generated on 2026-03-04 02:47 UTC*

> **Methodology**: All suites use `enable_thinking: false`. Cache cleared between suites. See [README](README.md) for details.

## Comparison Table

| Model | Quant | Hardware | Decode (s) | Decode (l) | Tools | Coding | Reasoning | General | Parser | Date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GLM-4.7-Flash-8bit | 8bit | Apple M3 Ultra (256GB) | 33.6 t/s | 57.9 t/s | 73% | 100% | 90% | 50% | glm47 | 2026-03-04 |
| GPT-OSS-20B-mxfp4-q8 | mxfp4+q8 | Apple M3 Ultra (256GB) | 97.4 t/s | 10.6 t/s | 17% | 60% | 20% | 90% | minimax | 2026-03-04 |
| Hermes-3-Llama-3.1-8B-4bit | 4bit | Apple M3 Ultra (256GB) | 71.9 t/s | 123.2 t/s | 17% | 20% | 30% | 40% | hermes | 2026-03-04 |
| MiniMax-M2.5-4bit | 4bit | Apple M3 Ultra (256GB) | 46.7 t/s | 50.5 t/s | 87% | 10% | 80% | 90% | minimax | 2026-03-04 |
| Qwen3-0.6B-4bit | 4bit | Apple M3 Ultra (256GB) | 78.4 t/s | 370.4 t/s | 30% | 20% | 20% | 30% | hermes | 2026-03-04 |
| Qwen3-Coder-Next-4bit | 4bit | Apple M3 Ultra (256GB) | 41.5 t/s | 73.6 t/s | 90% | 90% | 70% | 70% | hermes | 2026-03-04 |
| Qwen3-Coder-Next-6bit | 6bit | Apple M3 Ultra (256GB) | 34.8 t/s | 66.8 t/s | 87% | 90% | 80% | 70% | hermes | 2026-03-04 |
| Qwen3.5-122B-A10B-8bit | 8bit | Apple M3 Ultra (256GB) | 19.8 t/s | 42.9 t/s | 87% | 90% | 90% | 90% | hermes | 2026-03-04 |
| Qwen3.5-122B-A10B-mxfp4 | mxfp4 | Apple M3 Ultra (256GB) | 26.7 t/s | 56.9 t/s | 90% | 90% | 80% | 90% | hermes | 2026-03-04 |
| Qwen3.5-35B-A3B-4bit | 4bit | Apple M3 Ultra (256GB) | 36.2 t/s | 103.9 t/s | 87% | 90% | 50% | 70% | hermes | 2026-03-04 |
| Qwen3.5-35B-A3B-8bit | 8bit | Apple M3 Ultra (256GB) | 32.9 t/s | 81.6 t/s | 90% | 90% | 80% | 80% | hermes | 2026-03-04 |

## Details

### GLM-4.7-Flash-8bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: glm47
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser glm47`
- **Date**: 2026-03-04
- **TTFT**: cold=0.4s, warm=0.1s
- **Decode**: short=33.6 t/s, long=57.9 t/s
- **Tool Calling**: 73% (22/30)
- **Coding**: 100% (10/10)
- **Reasoning**: 90% (9/10)
- **General**: 50% (5/10)
- **Eval time**: 228.9s

### GPT-OSS-20B-mxfp4-q8

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: minimax
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser minimax`
- **Date**: 2026-03-04
- **TTFT**: cold=0.3s, warm=0.1s
- **Decode**: short=97.4 t/s, long=10.6 t/s
- **Tool Calling**: 17% (5/30)
- **Coding**: 60% (6/10)
- **Reasoning**: 20% (2/10)
- **General**: 90% (9/10)
- **Eval time**: 172.6s

### Hermes-3-Llama-3.1-8B-4bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-04
- **TTFT**: cold=0.1s, warm=0.1s
- **Decode**: short=71.9 t/s, long=123.2 t/s
- **Tool Calling**: 17% (5/30)
- **Coding**: 20% (2/10)
- **Reasoning**: 30% (3/10)
- **General**: 40% (4/10)
- **Eval time**: 111.4s

### MiniMax-M2.5-4bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: minimax
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser minimax`
- **Date**: 2026-03-04
- **TTFT**: cold=1.3s, warm=0.1s
- **Decode**: short=46.7 t/s, long=50.5 t/s
- **Tool Calling**: 87% (26/30)
- **Coding**: 10% (1/10)
- **Reasoning**: 80% (8/10)
- **General**: 90% (9/10)
- **Eval time**: 603.0s

### Qwen3-0.6B-4bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-04
- **TTFT**: cold=0.1s, warm=0.1s
- **Decode**: short=78.4 t/s, long=370.4 t/s
- **Tool Calling**: 30% (9/30)
- **Coding**: 20% (2/10)
- **Reasoning**: 20% (2/10)
- **General**: 30% (3/10)
- **Eval time**: 38.5s

### Qwen3-Coder-Next-4bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-04
- **TTFT**: cold=0.5s, warm=0.0s
- **Decode**: short=41.5 t/s, long=73.6 t/s
- **Tool Calling**: 90% (27/30)
- **Coding**: 90% (9/10)
- **Reasoning**: 70% (7/10)
- **General**: 70% (7/10)
- **Eval time**: 216.6s

### Qwen3-Coder-Next-6bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-04
- **TTFT**: cold=0.6s, warm=0.0s
- **Decode**: short=34.8 t/s, long=66.8 t/s
- **Tool Calling**: 87% (26/30)
- **Coding**: 90% (9/10)
- **Reasoning**: 80% (8/10)
- **General**: 70% (7/10)
- **Eval time**: 248.5s

### Qwen3.5-122B-A10B-8bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-04
- **TTFT**: cold=1.7s, warm=0.0s
- **Decode**: short=19.8 t/s, long=42.9 t/s
- **Tool Calling**: 87% (26/30)
- **Coding**: 90% (9/10)
- **Reasoning**: 90% (9/10)
- **General**: 90% (9/10)
- **Eval time**: 341.5s

### Qwen3.5-122B-A10B-mxfp4

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-04
- **TTFT**: cold=0.7s, warm=0.0s
- **Decode**: short=26.7 t/s, long=56.9 t/s
- **Tool Calling**: 90% (27/30)
- **Coding**: 90% (9/10)
- **Reasoning**: 80% (8/10)
- **General**: 90% (9/10)
- **Eval time**: 261.9s

### Qwen3.5-35B-A3B-4bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-04
- **TTFT**: cold=0.3s, warm=0.0s
- **Decode**: short=36.2 t/s, long=103.9 t/s
- **Tool Calling**: 87% (26/30)
- **Coding**: 90% (9/10)
- **Reasoning**: 50% (5/10)
- **General**: 70% (7/10)
- **Eval time**: 163.4s

### Qwen3.5-35B-A3B-8bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-04
- **TTFT**: cold=0.5s, warm=0.0s
- **Decode**: short=32.9 t/s, long=81.6 t/s
- **Tool Calling**: 90% (27/30)
- **Coding**: 90% (9/10)
- **Reasoning**: 80% (8/10)
- **General**: 80% (8/10)
- **Eval time**: 183.5s

---

## How to Add Your Results

1. Start vllm-mlx with your model: `vllm-mlx serve <model> --port 8000`
2. Run the eval: `python evals/run_eval.py --model "<model-name>" --quantization <quant>`
3. Your results are saved to `evals/results/<model>.json`
4. Regenerate this table: `python evals/generate_scorecard.py`
5. Submit a PR with your JSON file!

