# SPDX-License-Identifier: Apache-2.0
"""PR validation pipeline.

Single-command harness that grades an incoming PR for merge-readiness.
Each step is a discrete module under ``steps/``; the runner drives them
in a fixed order and the scorecard renders a strict pass/fail verdict.

Entry point::

    python3.12 -m scripts.pr_validate <PR#>

See ``scripts/pr_validate/README.md`` for the full step list and design.
"""
