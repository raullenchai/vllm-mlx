# vllm-mlx Model Scorecard

*Auto-generated on 2026-03-04 01:41 UTC*

> **Methodology**: General suite uses `enable_thinking: false`; other suites allow thinking. Cache cleared between suites. See [README](README.md) for details.

## Comparison Table

| Model | Quant | Hardware | Decode (s) | Decode (l) | Tools | Coding | Reasoning | General | Parser | Date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GLM-4.7-Flash-8bit | 8bit | Apple M3 Ultra (256GB) | 57.1 t/s | 57.1 t/s | 73% | 10% | 40% | 50% | glm47 | 2026-03-04 |
| GPT-OSS-20B-mxfp4-q8 | mxfp4+q8 | Apple M3 Ultra (256GB) | 98.3 t/s | 10.7 t/s | 17% | 60% | 20% | 90% | minimax | 2026-03-04 |
| Hermes-3-Llama-3.1-8B-4bit | 4bit | Apple M3 Ultra (256GB) | 71.1 t/s | 122.3 t/s | 17% | 20% | 30% | 40% | hermes | 2026-03-04 |
| MiniMax-M2.5-4bit | 4bit | Apple M3 Ultra (256GB) | 47 t/s | 51 t/s | 83% | 10% | 80% | 90% | minimax | 2026-03-04 |
| Qwen3-0.6B-4bit | 4bit | Apple M3 Ultra (256GB) | 290.4 t/s | 372.6 t/s | 50% | 0% | 10% | 30% | hermes | 2026-03-04 |
| Qwen3-Coder-Next-4bit | 4bit | Apple M3 Ultra (256GB) | 40.8 t/s | 73.1 t/s | 90% | 90% | 70% | 70% | hermes | 2026-03-04 |
| Qwen3-Coder-Next-6bit | 6bit | Apple M3 Ultra (256GB) | 35 t/s | 66.7 t/s | 87% | 90% | 80% | 70% | hermes | 2026-03-04 |
| Qwen3.5-122B-A10B-8bit | 8bit | Apple M3 Ultra (256GB) | 39.5 t/s | 42.9 t/s | 97% | 80% | 20% | 90% | hermes | 2026-03-04 |
| Qwen3.5-122B-A10B-mxfp4 | mxfp4 | Apple M3 Ultra (256GB) | 52.1 t/s | 57.5 t/s | 93% | 80% | 30% | 90% | hermes | 2026-03-04 |
| Qwen3.5-35B-A3B-4bit | 4bit | Apple M3 Ultra (256GB) | 90.8 t/s | 103.7 t/s | 90% | 10% | 30% | 70% | hermes | 2026-03-04 |
| Qwen3.5-35B-A3B-8bit | 8bit | Apple M3 Ultra (256GB) | 73 t/s | 81.1 t/s | 87% | 20% | 10% | 80% | hermes | 2026-03-04 |

## Details

### GLM-4.7-Flash-8bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: glm47
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser glm47`
- **Date**: 2026-03-04
- **TTFT**: cold=0.4s, warm=0.1s
- **Decode**: short=57.1 t/s, long=57.1 t/s
- **Tool Calling**: 73% (22/30)
- **Coding**: 10% (1/10)
- **Reasoning**: 40% (4/10)
- **General**: 50% (5/10)
- **Eval time**: 499.6s

### GPT-OSS-20B-mxfp4-q8

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: minimax
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser minimax`
- **Date**: 2026-03-04
- **TTFT**: cold=0.3s, warm=0.1s
- **Decode**: short=98.3 t/s, long=10.7 t/s
- **Tool Calling**: 17% (5/30)
- **Coding**: 60% (6/10)
- **Reasoning**: 20% (2/10)
- **General**: 90% (9/10)
- **Eval time**: 171.5s

### Hermes-3-Llama-3.1-8B-4bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-04
- **TTFT**: cold=0.2s, warm=0.1s
- **Decode**: short=71.1 t/s, long=122.3 t/s
- **Tool Calling**: 17% (5/30)
- **Coding**: 20% (2/10)
- **Reasoning**: 30% (3/10)
- **General**: 40% (4/10)
- **Eval time**: 111.9s

### MiniMax-M2.5-4bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: minimax
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser minimax`
- **Date**: 2026-03-04
- **TTFT**: cold=1.3s, warm=0.1s
- **Decode**: short=47 t/s, long=51 t/s
- **Tool Calling**: 83% (25/30)
- **Coding**: 10% (1/10)
- **Reasoning**: 80% (8/10)
- **General**: 90% (9/10)
- **Eval time**: 605.4s

### Qwen3-0.6B-4bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-04
- **TTFT**: cold=0.1s, warm=0.1s
- **Decode**: short=290.4 t/s, long=372.6 t/s
- **Tool Calling**: 50% (15/30)
- **Coding**: 0% (0/10)
- **Reasoning**: 10% (1/10)
- **General**: 30% (3/10)
- **Eval time**: 96.7s

### Qwen3-Coder-Next-4bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-04
- **TTFT**: cold=0.5s, warm=0.0s
- **Decode**: short=40.8 t/s, long=73.1 t/s
- **Tool Calling**: 90% (27/30)
- **Coding**: 90% (9/10)
- **Reasoning**: 70% (7/10)
- **General**: 70% (7/10)
- **Eval time**: 216.7s

### Qwen3-Coder-Next-6bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-04
- **TTFT**: cold=0.6s, warm=0.0s
- **Decode**: short=35 t/s, long=66.7 t/s
- **Tool Calling**: 87% (26/30)
- **Coding**: 90% (9/10)
- **Reasoning**: 80% (8/10)
- **General**: 70% (7/10)
- **Eval time**: 249.0s

### Qwen3.5-122B-A10B-8bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-04
- **TTFT**: cold=1.8s, warm=0.0s
- **Decode**: short=39.5 t/s, long=42.9 t/s
- **Tool Calling**: 97% (29/30)
- **Coding**: 80% (8/10)
- **Reasoning**: 20% (2/10)
- **General**: 90% (9/10)
- **Eval time**: 617.3s

### Qwen3.5-122B-A10B-mxfp4

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-04
- **TTFT**: cold=0.7s, warm=0.0s
- **Decode**: short=52.1 t/s, long=57.5 t/s
- **Tool Calling**: 93% (28/30)
- **Coding**: 80% (8/10)
- **Reasoning**: 30% (3/10)
- **General**: 90% (9/10)
- **Eval time**: 442.5s

### Qwen3.5-35B-A3B-4bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-04
- **TTFT**: cold=0.3s, warm=0.0s
- **Decode**: short=90.8 t/s, long=103.7 t/s
- **Tool Calling**: 90% (27/30)
- **Coding**: 10% (1/10)
- **Reasoning**: 30% (3/10)
- **General**: 70% (7/10)
- **Eval time**: 260.5s

### Qwen3.5-35B-A3B-8bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-04
- **TTFT**: cold=0.5s, warm=0.0s
- **Decode**: short=73 t/s, long=81.1 t/s
- **Tool Calling**: 87% (26/30)
- **Coding**: 20% (2/10)
- **Reasoning**: 10% (1/10)
- **General**: 80% (8/10)
- **Eval time**: 331.3s

---

## How to Add Your Results

1. Start vllm-mlx with your model: `vllm-mlx serve <model> --port 8000`
2. Run the eval: `python evals/run_eval.py --model "<model-name>" --quantization <quant>`
3. Your results are saved to `evals/results/<model>.json`
4. Regenerate this table: `python evals/generate_scorecard.py`
5. Submit a PR with your JSON file!

