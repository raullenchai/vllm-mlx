# SPDX-License-Identifier: Apache-2.0
"""``python3.12 -m scripts.pr_validate <PR#>`` entry point."""

from __future__ import annotations

import argparse
import sys

from .runner import run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pr_validate",
        description="Run the merge-readiness pipeline against a PR.",
    )
    parser.add_argument(
        "pr_number",
        type=int,
        help="GitHub PR number (e.g. 200)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print step output as it runs",
    )
    args = parser.parse_args(argv)
    return run_pipeline(args.pr_number, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
