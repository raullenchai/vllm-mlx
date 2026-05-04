# SPDX-License-Identifier: Apache-2.0
"""Step 6 — DeepSeek V4 Pro adversarial review of the diff.

Productionizes the one-off /tmp/deepseek_review_*.py scripts we've been
building per-PR. Reads the prompt template from
``scripts/pr_validate/prompts/deepseek_review.md``, sends the diff,
parses the response. Findings are surfaced in the scorecard verbatim
so a maintainer can decide whether to act on each.

Failure policy is deliberately conservative: we mark the step ``fail``
ONLY if DeepSeek's reply contains markers that look like blocking
findings (lines starting with "1.", "2.", … and mentioning specific
files). A reply of "No blocking issues found." → ``pass``. Network
failures or a missing API key → ``skip`` with a clear summary, NOT
``fail`` — we don't want a temporarily-down API to block every PR.
The strictness is at the human-review layer, not the API layer.

The API key is read from the environment (``DEEPSEEK_API_KEY``) and
falls back to a hardcoded development key documented in
``memory/knowledge/deepseek_api_key.md``. Putting it in source is OK
here ONLY because the key is the user's personal review-budget key,
not a production credential — but env override is preferred.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from ..base import Step, StepResult
from ..context import Context, env_truthy

# NB: do NOT default the API key here. The repo is public; even a
# personal review-budget key should never live in version control.
# Users provide it via ``DEEPSEEK_API_KEY`` (see
# ``memory/knowledge/deepseek_api_key.md`` for where the key is stored
# locally). Without the env var the step skips gracefully.
ENDPOINT = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-v4-pro"

# Hard cap on diff size we send. DeepSeek can take more, but past
# ~80KB of diff the signal-to-noise of the review drops sharply (the
# model starts skimming). For very large PRs we send the diff
# truncated and note it in the prompt.
MAX_DIFF_BYTES = 80_000

# Token budget. Reasoning-model behavior: we observed ~6K tokens of
# reasoning + 500-1500 tokens of visible content for a normal review.
# 16K leaves headroom.
MAX_TOKENS = 16_384

# How long we wait for the API. DeepSeek V4 Pro reasoning takes
# 30-90s typical, up to 5min for big diffs.
TIMEOUT_SECONDS = 600

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "deepseek_review.md"


class DeepSeekReviewStep(Step):
    name = "deepseek_review"
    description = "DeepSeek V4 Pro adversarial review of diff"

    def should_run(self, ctx: Context) -> bool:
        # Allow opt-out (CI without API access, offline dev, etc.).
        if env_truthy("PR_VALIDATE_NO_DEEPSEEK"):
            return False
        # Skip if there's no diff to review (shouldn't happen post-fetch
        # but defend against it — empty review wastes API budget).
        return bool(ctx.diff_path) and Path(ctx.diff_path).stat().st_size > 0

    def run(self, ctx: Context) -> StepResult:
        # httpx is in our deps already; importing inside the method
        # keeps the framework import-light for steps that don't need it.
        try:
            import httpx
        except ImportError:
            return StepResult(
                name=self.name,
                status="skip",
                summary="httpx not installed — `pip install httpx`",
            )

        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            return StepResult(
                name=self.name,
                status="skip",
                summary=(
                    "no DEEPSEEK_API_KEY set (export it to enable adversarial "
                    "review; see memory/knowledge/deepseek_api_key.md)"
                ),
            )

        diff = Path(ctx.diff_path).read_text()
        truncated = False
        if len(diff) > MAX_DIFF_BYTES:
            diff = diff[:MAX_DIFF_BYTES]
            truncated = True

        if not PROMPT_PATH.exists():
            return StepResult(
                name=self.name,
                status="error",
                summary=f"prompt template missing at {PROMPT_PATH}",
            )
        system_prompt = PROMPT_PATH.read_text()

        user_prompt = _build_user_prompt(ctx, diff, truncated)

        # Save what we sent — useful for debugging "why did the review
        # say X" without re-running.
        sent_path = ctx.artifact_path("deepseek-request.txt")
        sent_path.write_text(
            f"=== SYSTEM ===\n{system_prompt}\n\n=== USER ===\n{user_prompt}"
        )

        ctx.run_log(f"calling {MODEL} ({len(diff)} bytes of diff)…")

        try:
            with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
                resp = client.post(
                    ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": MODEL,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.0,
                        "max_tokens": MAX_TOKENS,
                    },
                )
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            return StepResult(
                name=self.name,
                status="skip",
                summary=f"API unreachable: {type(e).__name__}",
            )

        if resp.status_code != 200:
            return StepResult(
                name=self.name,
                status="skip",
                summary=f"API returned {resp.status_code}",
                details=f"```\n{resp.text[:1000]}\n```",
            )

        body = resp.json()
        choices = body.get("choices") or []
        if not choices:
            # DeepSeek can return an empty choices array on rate-limit
            # or content-policy refusal. Don't crash — surface as skip
            # with the body for debugging.
            return StepResult(
                name=self.name,
                status="skip",
                summary="API returned no choices (rate limit? policy?)",
                details=f"```json\n{json.dumps(body, indent=2)[:1000]}\n```",
            )
        content = choices[0].get("message", {}).get("content", "") or ""
        usage = body.get("usage", {})

        review_path = ctx.artifact_path("deepseek-review.md")
        review_path.write_text(content)
        usage_path = ctx.artifact_path("deepseek-usage.json")
        usage_path.write_text(json.dumps(usage, indent=2))

        # Parse the response. The prompt asks for a numbered list; we
        # detect "no findings" by an explicit phrase, otherwise we
        # extract numbered items as findings.
        findings = _extract_findings(content)
        no_issues = _is_clean_review(content)

        if no_issues and not findings:
            return StepResult(
                name=self.name,
                status="pass",
                summary="DeepSeek found no blocking issues",
                artifacts=[str(review_path), str(usage_path)],
            )

        # Findings present → fail the step in strict mode. Maintainer
        # decides per-finding whether to act; the scorecard surfaces
        # them all in the details. (Some reviewers may want a
        # human-decides flow here; that's what looking at the artifact
        # is for. The step's role is to surface, not to triage.)
        summary = f"{len(findings)} finding(s)"
        if truncated:
            summary += " (diff truncated for review)"
        return StepResult(
            name=self.name,
            status="fail",
            summary=summary,
            findings=findings,
            details=(
                "**Full review:**\n\n"
                f"{content}\n\n"
                f"_(Saved to `{review_path}`. "
                f"Token usage: {usage.get('total_tokens', '?')})_"
            ),
            artifacts=[str(review_path), str(usage_path)],
        )


def _build_user_prompt(ctx: Context, diff: str, truncated: bool) -> str:
    """Compose the user message: PR context + diff. The system prompt
    explains how to review; this provides what to review."""
    lines = [
        f"# PR #{ctx.pr_number}: {ctx.pr_title}",
        "",
        f"**Author**: {ctx.pr_author}{' (external/fork)' if ctx.pr_is_external else ''}",
        f"**Files**: {len(ctx.files_changed)} ({ctx.additions}+/{ctx.deletions}-)",
        f"**Blast radius**: {ctx.blast_radius}",
        "",
        "## Description",
        "",
        ctx.pr_body or "_(no description)_",
        "",
        "## Diff",
        "",
    ]
    if truncated:
        lines.append(
            f"_Note: diff truncated to {MAX_DIFF_BYTES} bytes for review. "
            "Full diff is on disk; review what's shown._"
        )
        lines.append("")
    lines.append("```diff")
    lines.append(diff)
    lines.append("```")
    return "\n".join(lines)


# Phrases that indicate a clean review. Match case-insensitively. If
# any of these appear in the response and we extract zero numbered
# findings, treat it as a clean pass.
_CLEAN_PATTERNS = (
    re.compile(r"no\s+blocking\s+issues?\s+found", re.IGNORECASE),
    re.compile(r"no\s+issues?\s+found", re.IGNORECASE),
    re.compile(r"^\s*looks?\s+good", re.IGNORECASE | re.MULTILINE),
)


def _is_clean_review(text: str) -> bool:
    return any(p.search(text) for p in _CLEAN_PATTERNS)


# Extract numbered findings — match lines starting with "1.", "2.",
# etc. (with optional leading whitespace and bold). One short line
# per finding for the scorecard table.
_FINDING_RE = re.compile(
    r"^\s*(?:\*\*)?(\d+)\.?\)?\s*(?:\*\*)?\s+(.+?)(?:\*\*)?\s*$",
    re.MULTILINE,
)


def _extract_findings(text: str) -> list[str]:
    """Pull numbered list items as findings. Truncates each to a
    reasonable length for the scorecard table; full text lives in the
    artifact file."""
    findings = []
    for match in _FINDING_RE.finditer(text):
        body = match.group(2).strip().rstrip("*").strip()
        # Cap per-finding length; long ones go in the artifact.
        if len(body) > 240:
            body = body[:237] + "…"
        findings.append(body)
    # Sometimes the model returns markdown headings like "### 1. Title"
    # which the regex above catches via the leading number; that's
    # fine. Deduplicate just in case (very long replies sometimes
    # repeat the summary at the bottom).
    seen = set()
    out = []
    for f in findings:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out
