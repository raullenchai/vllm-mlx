# Rapid DDTree Performance TODO

- [x] Skip per-cycle tokenizer decode for non-stream DDTree.
- [x] Add direct non-stream DDTree fast path in `DFlashEngine.generate`.
- [x] Make DDTree prefix-cache capture opt-in for benchmark/perf parity.
- [x] Fix DDTree fast-path ratio accounting.
- [x] Prefer engine-reported generation TPS in TUI.
- [x] Compile and run focused unit/synthetic checks.
- [x] Benchmark Rapid base vs Rapid DDTree after changes.
- [x] Commit and push.
