# DDTree Rapid-MLX Optimization TODO

## P0

- [x] Implement real DDTree streaming by accepted cycle/block instead of one final chunk.
- [x] Add cooperative cancellation for DDTree requests using `should_stop`.
- [x] Remove global n-gram disable for tool requests; rely on per-cycle tool guard.
- [x] Improve DDTree prefix cache with hit metadata, safe cloning, and status fields.
- [x] Propagate `prefix_boundary` into DDTree and store boundary prefill states.
- [x] Store prompt+generated-output cache state when safe for multi-turn agent reuse.
- [x] Document DDTree concurrency limitation and add safe overflow/queue behavior until real DDTree batching exists.

## P1

- [x] Pass scheduler/cache-related config into `DFlashEngine` or mirror useful fields.
- [x] Make DDTree non-greedy requests explicit: reject/fallback instead of silently ignoring sampler.
- [x] Use same Qwen tokenizer/EOS loading behavior as base engine.
- [x] Replace DDTree stop-string hot-loop decode with rolling suffix/token-aware matching.
- [x] Surface DDTree cache hit type and cached token count in active stats.

## P2

- [x] Default DDTree tree-build logprob normalization to approximate top-k unless exact is requested.
- [x] Reduce DDTree posterior CPU sync by transferring token ids instead of full logits where possible.
- [x] Add periodic Metal cache pressure cleanup for long DDTree generations.
- [x] Avoid metrics polling forcing expensive MLX memory stats during active DDTree generation.
- [x] Add config/benchmark notes for drafter-target matching, adaptive block sizing, and draft cache env vars.

## Verification

- [x] Python compile passes for touched files.
- [x] DDTree prefix cache smoke test passes.
- [x] Focused tests run or blocker documented.
