# SPDX-License-Identifier: Apache-2.0
"""User-facing doctor checks — lightweight, no dev tools required.

These checks run from a pip install without source checkout, pytest, or ruff.
They verify that the Rapid-MLX installation is functional.
"""

from __future__ import annotations

import time

from ..runner import CheckResult, Status, python_executable, run_subprocess


def check_metal() -> CheckResult:
    """Verify Apple Silicon Metal GPU is available."""
    t0 = time.perf_counter()
    py = python_executable()
    code = (
        "import mlx.core as mx; "
        "a = mx.ones(3); mx.eval(a); "
        "print(f'MLX OK: default device={mx.default_device()}')"
    )
    rc, stdout, stderr = run_subprocess([py, "-c", code], timeout=30)
    elapsed = time.perf_counter() - t0
    if rc == 0:
        return CheckResult(
            name="metal",
            status=Status.PASS,
            duration_s=elapsed,
            detail=stdout.strip(),
        )
    return CheckResult(
        name="metal",
        status=Status.FAIL,
        duration_s=elapsed,
        detail=f"MLX/Metal not available: {stderr[-500:]}",
    )


def check_imports() -> CheckResult:
    """Verify core modules import cleanly."""
    t0 = time.perf_counter()
    py = python_executable()
    code = (
        "import vllm_mlx; "
        "from vllm_mlx.api.models import ChatCompletionRequest; "
        "from vllm_mlx.engine import BatchedEngine; "
        "from vllm_mlx.agents import list_profiles; "
        "profiles = list_profiles(); "
        "print(f'OK: {len(profiles)} agent profiles loaded')"
    )
    rc, stdout, stderr = run_subprocess([py, "-c", code], timeout=30)
    elapsed = time.perf_counter() - t0
    if rc == 0:
        return CheckResult(
            name="imports",
            status=Status.PASS,
            duration_s=elapsed,
            detail=stdout.strip(),
        )
    return CheckResult(
        name="imports",
        status=Status.FAIL,
        duration_s=elapsed,
        detail=stderr[-500:],
    )


def check_cli() -> CheckResult:
    """Verify CLI commands respond."""
    t0 = time.perf_counter()
    py = python_executable()
    sub_cmds = [
        [py, "-m", "vllm_mlx.cli", "--help"],
        [py, "-m", "vllm_mlx.cli", "models"],
        [py, "-m", "vllm_mlx.cli", "agents"],
    ]
    failures: list[str] = []
    for sub in sub_cmds:
        rc, stdout, stderr = run_subprocess(sub, timeout=15)
        if rc != 0:
            failures.append(f"{' '.join(sub[-2:])} failed: {stderr[-200:]}")
    elapsed = time.perf_counter() - t0
    if not failures:
        return CheckResult(
            name="cli",
            status=Status.PASS,
            duration_s=elapsed,
            detail=f"{len(sub_cmds)} subcommands OK",
        )
    return CheckResult(
        name="cli",
        status=Status.FAIL,
        duration_s=elapsed,
        detail="\n".join(failures),
    )


def check_model_load() -> CheckResult:
    """Quick model load test with smallest available model."""
    import os
    import tempfile
    import textwrap

    t0 = time.perf_counter()
    py = python_executable()

    script = textwrap.dedent("""\
        import asyncio
        from vllm_mlx.model_aliases import resolve_model
        from vllm_mlx.engine import BatchedEngine

        async def test():
            model = resolve_model("qwen3-0.6b")
            e = BatchedEngine(model)
            await e.start()
            out = None
            async for o in e.stream_chat(
                messages=[{"role": "user", "content": "hi"}], max_tokens=3
            ):
                out = o
            await e.stop()
            if out is None:
                print("OK: model loaded (0 tokens generated)")
            else:
                print(f"OK: generated {out.completion_tokens} tokens")

        asyncio.run(test())
    """)

    # Write to temp file to avoid shell quoting issues
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(script)
        rc, stdout, stderr = run_subprocess([py, path], timeout=120)
    finally:
        os.unlink(path)

    elapsed = time.perf_counter() - t0
    if rc == 0:
        return CheckResult(
            name="model_load",
            status=Status.PASS,
            duration_s=elapsed,
            detail=stdout.strip(),
        )
    # Model download failure is expected — skip, don't fail
    if (
        "HTTP 404" in stderr
        or "Repository Not Found" in stderr
        or "does not appear" in stderr.lower()
    ):
        return CheckResult(
            name="model_load",
            status=Status.SKIP,
            duration_s=elapsed,
            detail="Test model not available (download required)",
        )
    return CheckResult(
        name="model_load",
        status=Status.FAIL,
        duration_s=elapsed,
        detail=stderr[-500:],
    )
