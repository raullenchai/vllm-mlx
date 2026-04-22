# SPDX-License-Identifier: Apache-2.0
"""Recipe engine: model × hardware → recommendation + command.

Core formula:
    available_mem = hardware.memory_gb - model.model_memory_gb - OS_OVERHEAD
    max_context   = available_mem / kv_per_token
    estimated_tps = model.bandwidth_efficiency * hardware.bandwidth_gbs / model_bytes_per_token
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .hardware import HardwareProfile
from .models import ModelRecipe

OS_OVERHEAD_GB = 4  # macOS + background apps


@dataclass
class Recommendation:
    model: ModelRecipe
    hardware: HardwareProfile

    # Computed
    fits: bool  # can the model even load?
    available_memory_gb: float
    max_context_tokens: int
    max_context_turbo_tokens: int | None  # with TurboQuant
    estimated_tps: float
    needs_turboquant: bool  # needs TurboQuant for reasonable context

    # Generated command
    command: str
    command_args: list[str] = field(default_factory=list)

    # Status
    status: str = ""  # "comfortable", "tight", "oom"
    notes: list[str] = field(default_factory=list)


def compute_recommendation(
    model: ModelRecipe,
    hardware: HardwareProfile,
    desired_context: int | None = None,
) -> Recommendation:
    """Compute hardware-specific recommendation for a model."""
    available_gb = hardware.memory_gb - model.model_memory_gb - OS_OVERHEAD_GB
    fits = available_gb > 0

    if not fits:
        return Recommendation(
            model=model,
            hardware=hardware,
            fits=False,
            available_memory_gb=available_gb,
            max_context_tokens=0,
            max_context_turbo_tokens=0,
            estimated_tps=0,
            needs_turboquant=False,
            command="",
            status="oom",
            notes=[
                f"Model needs {model.model_memory_gb:.0f}GB, "
                f"only {hardware.memory_gb - OS_OVERHEAD_GB:.0f}GB available"
            ],
        )

    # Max context without TurboQuant
    available_mb = available_gb * 1024
    kv_mb_per_token = model.kv_per_1k_tokens_mb / 1000
    max_ctx = int(available_mb / kv_mb_per_token) if kv_mb_per_token > 0 else 0

    # Max context with TurboQuant
    max_ctx_turbo = None
    if model.turboquant_compatible and model.kv_per_1k_turbo_mb:
        kv_turbo_per_token = model.kv_per_1k_turbo_mb / 1000
        max_ctx_turbo = (
            int(available_mb / kv_turbo_per_token) if kv_turbo_per_token > 0 else 0
        )

    # Estimate decode tok/s via bandwidth scaling
    estimated_tps = (
        model.bandwidth_efficiency
        * hardware.bandwidth_gbs
        / (
            model.model_memory_gb
            / 1  # ~1 byte per param at decode time for memory-bound models
        )
    )
    # But don't exceed measured_tps scaled by bandwidth ratio
    bw_ratio = hardware.bandwidth_gbs / 800  # 800 = M3 Ultra reference
    estimated_tps = model.measured_tps * bw_ratio

    # Determine if TurboQuant is needed
    target_ctx = desired_context or 32768
    needs_turbo = max_ctx < target_ctx and (
        max_ctx_turbo is not None and max_ctx_turbo >= target_ctx
    )

    # Determine status
    if max_ctx >= 65536:
        status = "comfortable"
    elif max_ctx >= 16384:
        status = "tight"
    else:
        status = "minimal"

    # Build command
    args = list(model.base_args)
    notes = []

    if needs_turbo:
        args.extend(["--kv-cache-turboquant"])
        effective_ctx = max_ctx_turbo
        notes.append("TurboQuant enabled for longer context")
    else:
        effective_ctx = max_ctx

    # Cap at reasonable powers of 2
    ctx_cap = _round_down_context(min(effective_ctx, 131072))
    if desired_context:
        ctx_cap = min(ctx_cap, desired_context)
    args.extend(["--max-kv-cache-tokens", str(ctx_cap)])

    command = _build_command(model.model_id, args)

    return Recommendation(
        model=model,
        hardware=hardware,
        fits=True,
        available_memory_gb=available_gb,
        max_context_tokens=max_ctx,
        max_context_turbo_tokens=max_ctx_turbo,
        estimated_tps=estimated_tps,
        needs_turboquant=needs_turbo,
        command=command,
        command_args=args,
        status=status,
        notes=notes,
    )


def _round_down_context(tokens: int) -> int:
    """Round down to nearest standard context size."""
    standard = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
    for s in reversed(standard):
        if tokens >= s:
            return s
    return 2048


def _build_command(model_id: str, args: list[str]) -> str:
    """Build the rapid-mlx serve command string."""
    parts = ["rapid-mlx", "serve", model_id]
    parts.extend(args)
    return " \\\n  ".join(_chunk_args(parts))


def _chunk_args(parts: list[str]) -> list[str]:
    """Group CLI args into flag+value pairs for line wrapping."""
    chunks = [parts[0] + " " + parts[1] + " " + parts[2]]  # rapid-mlx serve <model>
    i = 3
    while i < len(parts):
        if (
            parts[i].startswith("--")
            and i + 1 < len(parts)
            and not parts[i + 1].startswith("--")
        ):
            chunks.append(parts[i] + " " + parts[i + 1])
            i += 2
        else:
            chunks.append(parts[i])
            i += 1
    return chunks


def format_recommendation(rec: Recommendation) -> str:
    """Format a recommendation for terminal display."""
    lines = []
    m = rec.model
    h = rec.hardware

    # Header
    lines.append(f"  {m.name} on {h.name}")
    lines.append(f"  {'=' * (len(m.name) + 4 + len(h.name))}")
    lines.append("")

    if not rec.fits:
        lines.append("  Status: OOM")
        for n in rec.notes:
            lines.append(f"    {n}")
        return "\n".join(lines)

    # Specs
    lines.append(f"  Model:      {m.model_id}")
    lines.append(
        f"  Arch:       {m.architecture} ({m.parameter_count}{'/' + m.active_parameters + ' active' if m.active_parameters else ''})"
    )
    lines.append(f"  Quant:      {m.quantization}")
    lines.append(f"  Weights:    {m.model_memory_gb:.1f} GB")
    lines.append(f"  Available:  {rec.available_memory_gb:.0f} GB for KV cache")
    lines.append("")

    # Context
    lines.append(f"  Max context:         {rec.max_context_tokens:>7,} tokens")
    if rec.max_context_turbo_tokens:
        lines.append(
            f"  Max context (Turbo): {rec.max_context_turbo_tokens:>7,} tokens"
        )
    lines.append("")

    # Performance
    lines.append(f"  Est. decode:  ~{rec.estimated_tps:.0f} tok/s")
    lines.append(f"  Status:       {rec.status}")
    lines.append("")

    # Features
    features = []
    if m.tool_calling:
        features.append(f"tool calling ({m.tool_parser})")
    if m.reasoning:
        features.append(f"reasoning ({m.reasoning_parser})")
    if m.vision:
        features.append("vision")
    if m.turboquant_compatible:
        features.append("TurboQuant")
    lines.append(f"  Features:  {', '.join(features) if features else 'none'}")
    lines.append("")

    # Notes
    if rec.notes:
        for n in rec.notes:
            lines.append(f"  Note: {n}")
        lines.append("")
    if m.notes:
        lines.append(f"  Tip: {m.notes}")
        lines.append("")

    # Command
    lines.append("  Command:")
    lines.append(f"  $ {rec.command}")

    return "\n".join(lines)


def format_models_table(
    recipes: list[ModelRecipe],
    hardware: HardwareProfile | None = None,
) -> str:
    """Format a table of models, optionally with hardware-specific info."""
    lines = []

    if hardware:
        lines.append(
            f"  Models for {hardware.name} ({hardware.memory_gb}GB, {hardware.bandwidth_gbs} GB/s)"
        )
    else:
        lines.append(
            "  Available Models (use --hardware or auto-detect for specific recommendations)"
        )
    lines.append("")

    # Table header
    if hardware:
        lines.append(
            f"  {'ID':<22} {'Name':<30} {'RAM':<7} {'Ctx':<8} {'tok/s':<7} {'Tools':<6} {'Status'}"
        )
        lines.append(
            f"  {'─' * 22} {'─' * 30} {'─' * 7} {'─' * 8} {'─' * 7} {'─' * 6} {'─' * 10}"
        )
    else:
        lines.append(
            f"  {'ID':<22} {'Name':<30} {'RAM':<7} {'Quant':<6} {'Ref tok/s':<10} {'Tools':<6}"
        )
        lines.append(
            f"  {'─' * 22} {'─' * 30} {'─' * 7} {'─' * 6} {'─' * 10} {'─' * 6}"
        )

    for r in sorted(recipes, key=lambda x: x.model_memory_gb):
        tools = "Yes" if r.tool_calling else "No"
        if hardware:
            rec = compute_recommendation(r, hardware)
            if not rec.fits:
                ctx_str = "OOM"
                tps_str = "-"
                status = "OOM"
            else:
                ctx_str = f"{_round_down_context(rec.max_context_tokens) // 1024}K"
                tps_str = f"~{rec.estimated_tps:.0f}"
                status = rec.status
            lines.append(
                f"  {r.id:<22} {r.name:<30} {r.model_memory_gb:<7.1f} {ctx_str:<8} {tps_str:<7} {tools:<6} {status}"
            )
        else:
            lines.append(
                f"  {r.id:<22} {r.name:<30} {r.model_memory_gb:<7.1f} {r.quantization:<6} {r.measured_tps:<10.0f} {tools:<6}"
            )

    lines.append("")
    lines.append("  Use: rapid-mlx recipe <model-id> for detailed recommendation")
    return "\n".join(lines)
