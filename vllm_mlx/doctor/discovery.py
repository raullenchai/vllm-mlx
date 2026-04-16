# SPDX-License-Identifier: Apache-2.0
"""Detect which model aliases have weights present locally.

The benchmark tier uses this to skip aliases whose weights would
otherwise trigger a multi-GB download mid-run.  We deliberately don't
*download* anything — the user is expected to pre-fetch what they want
benchmarked.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .runner import REPO_ROOT


@dataclass
class ModelAvailability:
    alias: str
    repo_id: str
    available: bool
    path: Path | None = None
    reason: str = ""  # only set when available=False


def load_aliases() -> dict[str, str]:
    """Return the alias → repo-id mapping shipped with the package."""
    aliases_path = REPO_ROOT / "vllm_mlx" / "aliases.json"
    with open(aliases_path) as f:
        return json.load(f)


def _hf_cache_roots() -> list[Path]:
    """Return all candidate HF cache directories on this host.

    HF_HUB_CACHE wins (this is what the doctor sets in the spawned
    server's env).  ~/.cache/huggingface/hub is the standard fallback.
    LM Studio's cache is also checked because users often have models
    there from prior LM Studio installs.
    """
    roots: list[Path] = []
    env_cache = os.environ.get("HF_HUB_CACHE")
    if env_cache:
        roots.append(Path(env_cache))
    env_home = os.environ.get("HF_HOME")
    if env_home:
        roots.append(Path(env_home) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    roots.append(Path.home() / ".lmstudio" / "models")
    # Dedupe while preserving order.
    seen: set[Path] = set()
    out: list[Path] = []
    for r in roots:
        if r and r not in seen:
            out.append(r)
            seen.add(r)
    return out


def _repo_to_hf_dirname(repo_id: str) -> str:
    """HF cache uses 'models--{org}--{repo}' as the directory name."""
    return "models--" + repo_id.replace("/", "--")


def _check_alias(alias: str, repo_id: str, cache_roots: list[Path]) -> ModelAvailability:
    """Look for the model in the candidate cache roots."""
    target = _repo_to_hf_dirname(repo_id)
    for root in cache_roots:
        candidate = root / target
        if candidate.exists() and any(candidate.iterdir()):
            # Sanity: at least one snapshot must contain a config.json.
            snapshots = candidate / "snapshots"
            if snapshots.exists():
                for snap in snapshots.iterdir():
                    if (snap / "config.json").exists():
                        return ModelAvailability(
                            alias=alias, repo_id=repo_id,
                            available=True, path=snap,
                        )
            # LM Studio layout: just the model files in a subdir.
            if (candidate / "config.json").exists():
                return ModelAvailability(
                    alias=alias, repo_id=repo_id,
                    available=True, path=candidate,
                )
    return ModelAvailability(
        alias=alias, repo_id=repo_id, available=False,
        reason="not found in HF_HUB_CACHE / ~/.cache/huggingface / ~/.lmstudio",
    )


def discover_local_models() -> list[ModelAvailability]:
    """Return availability info for every alias in aliases.json.

    Sorted: available first (alphabetical), then unavailable.
    """
    aliases = load_aliases()
    cache_roots = _hf_cache_roots()
    results = [_check_alias(a, r, cache_roots) for a, r in aliases.items()]
    results.sort(key=lambda m: (not m.available, m.alias))
    return results


def available_aliases() -> list[str]:
    """Convenience: just the aliases that are present locally."""
    return [m.alias for m in discover_local_models() if m.available]
