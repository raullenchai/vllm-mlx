# PR validation pipeline

Single-command merge-readiness gate for incoming PRs (especially
external contributions). Strict mode: any single step failure blocks
merge.

## Usage

```bash
# from the repo root
python3.12 -m scripts.pr_validate <PR#>

# verbose mode (more progress logging on stderr)
python3.12 -m scripts.pr_validate <PR#> -v

# stdout = markdown scorecard (paste into PR comment)
# stderr = progress logs
# exit 0 = MERGE-SAFE, exit 1 = DO NOT MERGE
```

## Pipeline

| # | step | gate | runtime |
|---|---|---|---|
| 0 | `fetch` | always (fail-fast) | ~3s |
| 6 | `deepseek_review` | always (skip if no API) | 30–90s |
| 1 | `supply_chain` | always | ~5s |
| 2 | `lint` | when diff has .py | ~3s |
| 3 | `targeted_tests` | when diff has .py | 30s–3min |
| 4 | `full_unit` | blast ≥ medium | ~25s |
| 5 | `stress_e2e_bench` | blast == high | 5–10min |

(DeepSeek review goes second by design: get cheap critical thinking
*before* spending 10 minutes on tests.)

## Verdict

Strict — any single `fail` or `error` blocks merge. `skip` is neutral.

## Blast radius

Computed from `files_changed`. See `context.py::HIGH_BLAST_PATHS` for
the gating list. The classification chooses which expensive steps run.

* **high** — touches scheduler / engine / cli / server / memory_cache
  / routes / pyproject.toml. Full battery.
* **medium** — touches `vllm_mlx/` or `tests/` but not the high-blast
  list. Skips stress.
* **low** — only docs / examples / README. Skips full_unit + stress.

## Adding a step

1. Write a module under `steps/` with a class extending `base.Step`.
2. Set `name`, `description`, override `run(ctx)` (and `should_run` if
   the step is conditional).
3. Import + insert in the `STEPS` list in `runner.py`.

The runner orders steps explicitly — no auto-discovery — so the
pipeline policy is grep-able from one file.

## Step details

### `fetch` (step 0)

Wraps `gh pr view --json` + `gh pr diff`. Saves the diff to
`<work_dir>/pr.diff`. Refuses CLOSED / MERGED / DIRTY (merge-conflict)
PRs by design — re-open or rebase first.

### `deepseek_review` (step 6, runs second)

Sends the diff to DeepSeek V4 Pro with the prompt at
`prompts/deepseek_review.md`. Findings go in the scorecard. Skips if
`PR_VALIDATE_NO_DEEPSEEK=1` or no API key. Skips on network failure
(don't block PRs on a flaky API).

API key resolution: `$DEEPSEEK_API_KEY` → fallback to dev key in code
(see `memory/knowledge/deepseek_api_key.md`).

### `supply_chain` (step 1)

* Flags any modification to install hooks, CI workflows, Makefile,
  Homebrew tap (BLOCKING for external authors, warning for collaborators).
* Greps added lines for suspicious patterns (`eval`, `exec`,
  `pickle.loads`, `subprocess(... shell=True)`, hardcoded URLs/IPs,
  large hex/base64 blobs).
* Runs `pip-audit` against any new dependencies declared in
  `pyproject.toml` / `requirements.txt`.

### `lint` (step 2)

`ruff check` + `ruff format --check` on the changed `.py` files only.

### `targeted_tests` (step 3)

Maps each changed `.py` to candidate test files (heuristic by
filename) and runs them. **Negative control**: if any fail, re-runs
the same set on a fresh `git worktree` of `main` to filter
pre-existing flakes. Real regressions = fail.

### `full_unit` (step 4)

`pytest tests/` minus integrations + event_loop. Gated on blast ≥
medium (low-blast PRs can't break runtime).

### `stress_e2e_bench` (step 5)

The heaviest. Gated on blast == high. For each model in
`golden_models.yaml` that fits machine RAM (highest-quality candidate
per family):

1. Boot a server on port 8451.
2. Run `scripts/stress_test.py` (8 stress scenarios).
3. Run each agent integration in the registry (matrix: m × n).
4. Run an inline bench (cold TTFT + warm TTFT + speedup).
5. Compare bench to `harness/baselines/bench-<model>.json` —
   regression > 5% = fail.

Skip with `PR_VALIDATE_NO_STRESS=1`.

## Artifacts

Every run writes to `/tmp/pr_validate/pr-<N>/`:

* `pr.diff` — full unified diff
* `lint-{check,format}.log` — ruff output
* `targeted-{pr,main}.log` — pytest output (PR + neg-control on main)
* `full-unit.log` — pytest full output
* `deepseek-{request,review,usage}.{txt,md,json}` — API call + result
* `supply-chain-scan.log` + `pip-audit.log` — supply-chain artifacts
* `server-<model>.log` — boot + lifespan log
* `stress-<model>.log` — stress test output
* `agent-<name>-<model>.log` — per-integration output
* `bench-<model>.json` — bench numbers (latest run)

## Roadmap

* GitHub Action wiring (cheap layers only — DeepSeek per-push is
  expensive).
* License-drift check via `pip show <pkg>` against an allowlist.
* Diff-aware import-graph for `targeted_tests` (replace stem heuristic).
* Expand `golden_models.yaml` to the full family list once we have RAM
  budget for big-model boots.
