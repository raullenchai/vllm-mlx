.PHONY: help smoke check full benchmark update-baselines lint test clean

# Pick the interpreter:
#   1. Active venv ($VIRTUAL_ENV/bin/python) — wins so contributors using
#      a 3.10/3.11/3.13 venv get their venv's python regardless of PATH.
#   2. Versioned binaries that actually run a >=3.10 interpreter — we
#      must run --version because pyenv shims appear on PATH for *every*
#      version even if only one is installed, and macOS's bare 'python'
#      is often system 3.9 (below requires-python).
#   3. python3 last-resort fallback (lets the user see a clean error if
#      nothing on the system meets the version requirement).
# Override explicitly with: make smoke PY=python3.13
PY ?= $(shell \
  if [ -n "$$VIRTUAL_ENV" ] && [ -x "$$VIRTUAL_ENV/bin/python" ]; then \
    echo "$$VIRTUAL_ENV/bin/python"; exit 0; \
  fi; \
  for cand in python3.13 python3.12 python3.11 python3.10 python3; do \
    path=$$(command -v $$cand 2>/dev/null) || continue; \
    "$$path" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' \
      2>/dev/null && echo "$$path" && exit 0; \
  done; \
  echo python3)
HF_HUB_CACHE ?= $(shell echo $$HF_HUB_CACHE)
DOCTOR := $(PY) -m vllm_mlx.cli doctor

help:
	@echo "Rapid-MLX developer targets:"
	@echo ""
	@echo "  Doctor (regression harness — see harness/README.md):"
	@echo "    make smoke              ~2 min,  no model required"
	@echo "    make check              ~10 min, qwen3.5-4b"
	@echo "    make full               ~1-2 hr, 3 models + 11 agents"
	@echo "    make benchmark          overnight, all local models → scorecard"
	@echo "    make update-baselines TIER=check  re-record baseline (after review)"
	@echo ""
	@echo "  Quick checks:"
	@echo "    make lint               ruff over vllm_mlx/ + tests/"
	@echo "    make test               pytest unit suite (excludes integration)"
	@echo ""
	@echo "  Env: HF_HUB_CACHE=$(HF_HUB_CACHE)"

# ---------- doctor tiers ----------
smoke:
	$(DOCTOR) smoke

check:
	$(DOCTOR) check

full:
	$(DOCTOR) full

benchmark:
	$(DOCTOR) benchmark

# Usage: make update-baselines TIER=check
# Always inspect 'git diff harness/baselines/' before committing.
update-baselines:
	@if [ -z "$(TIER)" ]; then \
		echo "error: TIER is required. Example: make update-baselines TIER=check"; \
		exit 2; \
	fi
	$(DOCTOR) $(TIER) --update-baselines

# ---------- quick checks ----------
lint:
	@if $(PY) -m ruff --version >/dev/null 2>&1; then \
		$(PY) -m ruff check vllm_mlx/ tests/; \
	elif command -v ruff >/dev/null 2>&1; then \
		ruff check vllm_mlx/ tests/; \
	else \
		echo "ruff not installed — pip install ruff"; \
		exit 1; \
	fi

test:
	$(PY) -m pytest tests/ -q --ignore=tests/integrations \
	    --deselect tests/test_event_loop.py

clean:
	rm -rf harness/runs/*
	@echo "Cleared harness/runs/ — baselines kept."
