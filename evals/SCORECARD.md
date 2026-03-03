# vllm-mlx Model Scorecard

*Auto-generated on 2026-03-03 05:32 UTC*

> **Methodology**: All evals run with `enable_thinking: false` across all models for fair comparison. See [README](README.md) for details.

## Comparison Table

| Model | Quant | Hardware | Decode (s) | Decode (l) | Tools | Coding | Reasoning | General | Parser | Date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GLM-4.7-Flash-8bit | 8bit | Apple M3 Ultra (256GB) | 31.3 t/s | 58 t/s | 73% | 100% | 80% | 90% | glm47 | 2026-03-03 |
| GPT-OSS-20B-mxfp4-q8 | mxfp4-q8 | Apple M3 Ultra (256GB) | 89.1 t/s | 124 t/s | 77% | 90% | 90% | 100% | harmony | 2026-03-03 |
| Hermes-3-Llama-3.1-8B-4bit | 4bit | Apple M3 Ultra (256GB) | 70.5 t/s | 123.2 t/s | 17% | 100% | 70% | 60% | hermes | 2026-03-03 |
| MiniMax-M2.5-4bit | 4bit | Apple M3 Ultra (256GB) | 44.9 t/s | 50.6 t/s | 87% | 40% | 60% | 90% | minimax | 2026-03-03 |
| Qwen3-0.6B-4bit | 4bit | Apple M3 Ultra (256GB) | 293.8 t/s | 372.3 t/s | 50% | 0% | 30% | 50% | hermes | 2026-03-03 |
| Qwen3-Coder-Next-4bit | 4bit | Apple M3 Ultra (256GB) | 35.5 t/s | 73.9 t/s | 90% | 100% | 80% | 100% | hermes | 2026-03-03 |
| Qwen3-Coder-Next-6bit | 6bit | Apple M3 Ultra (256GB) | 33.1 t/s | 67.7 t/s | 87% | 100% | 90% | 100% | hermes | 2026-03-03 |
| Qwen3.5-122B-A10B-8bit | 8bit | Apple M3 Ultra (256GB) | 20.8 t/s | 43.9 t/s | 93% | 60% | 90% | 60% | hermes | 2026-03-03 |
| Qwen3.5-122B-A10B-mxfp4 | mxfp4 | Apple M3 Ultra (256GB) | 28.3 t/s | 57.7 t/s | 90% | 90% | 90% | 80% | hermes | 2026-03-03 |
| Qwen3.5-35B-A3B-4bit | 4bit | Apple M3 Ultra (256GB) | 38.6 t/s | 103.6 t/s | 77% | 100% | 90% | 80% | hermes | 2026-03-03 |
| Qwen3.5-35B-A3B-8bit | 8bit | Apple M3 Ultra (256GB) | 34.2 t/s | 81.5 t/s | 77% | 100% | 90% | 90% | hermes | 2026-03-03 |

## Details

### GLM-4.7-Flash-8bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: glm47
- **Date**: 2026-03-03
- **TTFT**: cold=1.1s, warm=0.4s
- **Decode**: short=31.3 t/s, long=58 t/s
- **Tool Calling**: 73% (22/30)
- **Coding**: 100% (10/10)
- **Reasoning**: 80% (8/10)
- **General**: 90% (9/10)
- **Eval time**: 130.7s

### GPT-OSS-20B-mxfp4-q8

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: harmony
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser harmony`
- **Date**: 2026-03-03
- **TTFT**: cold=0.4s, warm=0.2s
- **Decode**: short=89.1 t/s, long=124 t/s
- **Tool Calling**: 77% (23/30)
- **Coding**: 90% (9/10)
- **Reasoning**: 90% (9/10)
- **General**: 100% (10/10)
- **Eval time**: 98.5s

### Hermes-3-Llama-3.1-8B-4bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Date**: 2026-03-03
- **TTFT**: cold=0.4s, warm=0.2s
- **Decode**: short=70.5 t/s, long=123.2 t/s
- **Tool Calling**: 17% (5/30)
- **Coding**: 100% (10/10)
- **Reasoning**: 70% (7/10)
- **General**: 60% (6/10)
- **Eval time**: 79.6s

### MiniMax-M2.5-4bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: minimax
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser minimax`
- **Date**: 2026-03-03
- **TTFT**: cold=1.4s, warm=0.5s
- **Decode**: short=44.9 t/s, long=50.6 t/s
- **Tool Calling**: 87% (26/30)
- **Coding**: 40% (4/10)
- **Reasoning**: 60% (6/10)
- **General**: 90% (9/10)
- **Eval time**: 385.3s

### Qwen3-0.6B-4bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-03
- **TTFT**: cold=0.1s, warm=0.1s
- **Decode**: short=293.8 t/s, long=372.3 t/s
- **Tool Calling**: 50% (15/30)
- **Coding**: 0% (0/10)
- **Reasoning**: 30% (3/10)
- **General**: 50% (5/10)
- **Eval time**: 77.5s

### Qwen3-Coder-Next-4bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-03
- **TTFT**: cold=0.7s, warm=0.0s
- **Decode**: short=35.5 t/s, long=73.9 t/s
- **Tool Calling**: 90% (27/30)
- **Coding**: 100% (10/10)
- **Reasoning**: 80% (8/10)
- **General**: 100% (10/10)
- **Eval time**: 128.1s

### Qwen3-Coder-Next-6bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-03
- **TTFT**: cold=0.8s, warm=0.0s
- **Decode**: short=33.1 t/s, long=67.7 t/s
- **Tool Calling**: 87% (26/30)
- **Coding**: 100% (10/10)
- **Reasoning**: 90% (9/10)
- **General**: 100% (10/10)
- **Eval time**: 138.8s

### Qwen3.5-122B-A10B-8bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-03
- **TTFT**: cold=1.8s, warm=0.0s
- **Decode**: short=20.8 t/s, long=43.9 t/s
- **Tool Calling**: 93% (28/30)
- **Coding**: 60% (6/10)
- **Reasoning**: 90% (9/10)
- **General**: 60% (6/10)
- **Eval time**: 555.0s

### Qwen3.5-122B-A10B-mxfp4

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-03
- **TTFT**: cold=1.0s, warm=0.0s
- **Decode**: short=28.3 t/s, long=57.7 t/s
- **Tool Calling**: 90% (27/30)
- **Coding**: 90% (9/10)
- **Reasoning**: 90% (9/10)
- **General**: 80% (8/10)
- **Eval time**: 233.6s

### Qwen3.5-35B-A3B-4bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-03
- **TTFT**: cold=0.5s, warm=0.0s
- **Decode**: short=38.6 t/s, long=103.6 t/s
- **Tool Calling**: 77% (23/30)
- **Coding**: 100% (10/10)
- **Reasoning**: 90% (9/10)
- **General**: 80% (8/10)
- **Eval time**: 147.0s

### Qwen3.5-35B-A3B-8bit

- **Hardware**: Apple M3 Ultra (256GB)
- **Parser**: hermes
- **Server flags**: `--enable-auto-tool-choice --tool-call-parser hermes`
- **Date**: 2026-03-03
- **TTFT**: cold=0.7s, warm=0.0s
- **Decode**: short=34.2 t/s, long=81.5 t/s
- **Tool Calling**: 77% (23/30)
- **Coding**: 100% (10/10)
- **Reasoning**: 90% (9/10)
- **General**: 90% (9/10)
- **Eval time**: 108.0s

---

## How to Add Your Results

1. Start vllm-mlx with your model: `vllm-mlx serve <model> --port 8000`
2. Run the eval: `python evals/run_eval.py --model "<model-name>" --quantization <quant>`
3. Your results are saved to `evals/results/<model>.json`
4. Regenerate this table: `python evals/generate_scorecard.py`
5. Submit a PR with your JSON file!

