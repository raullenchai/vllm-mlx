# SPDX-License-Identifier: Apache-2.0
"""CLI entry points for ``rapid-mlx doctor``."""

from __future__ import annotations

import sys

from .runner import REPO_ROOT, DoctorRunner


def _require_source_checkout() -> None:
    """Doctor depends on tests/ + harness/ + pyproject.toml.

    A pip-installed wheel does not ship those, so the command would
    immediately produce confusing failures.  Detect that case up front
    and exit with a clear message instead.
    """
    sentinels = [
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "tests",
        REPO_ROOT / "harness",
    ]
    missing = [str(p.relative_to(REPO_ROOT)) for p in sentinels if not p.exists()]
    if missing:
        print(
            "[doctor] this command requires a source checkout of Rapid-MLX.\n"
            f"        missing: {', '.join(missing)}\n"
            "        Clone https://github.com/raullenchai/Rapid-MLX and run "
            "from the repo root.",
            file=sys.stderr,
        )
        sys.exit(2)


def doctor_command(args) -> None:
    """Dispatch to the requested tier."""
    _require_source_checkout()

    tier = getattr(args, "tier", None) or "smoke"

    if tier == "smoke":
        result = run_smoke_tier()
    elif tier == "check":
        print("[doctor] check tier not yet implemented", file=sys.stderr)
        sys.exit(2)
    elif tier == "full":
        print("[doctor] full tier not yet implemented", file=sys.stderr)
        sys.exit(2)
    elif tier == "benchmark":
        print("[doctor] benchmark tier not yet implemented", file=sys.stderr)
        sys.exit(2)
    else:
        print(f"[doctor] unknown tier: {tier}", file=sys.stderr)
        sys.exit(2)

    sys.exit(result.exit_code)


def run_smoke_tier():
    """Static + import + CLI sanity. No model required."""
    from .checks import smoke

    print("Rapid-MLX Doctor — smoke tier")
    print("=" * 60)

    runner = DoctorRunner(tier="smoke")
    runner.run_check("repo_layout", smoke.check_repo_layout)
    runner.run_check("imports", smoke.check_imports)
    runner.run_check("ruff", smoke.check_ruff)
    runner.run_check("cli_sanity", smoke.check_cli_sanity)
    runner.run_check("pytest", smoke.check_pytest_unit)
    return runner.finalize()
