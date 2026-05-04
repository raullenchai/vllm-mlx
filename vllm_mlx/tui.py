# SPDX-License-Identifier: Apache-2.0
"""Small live monitor for `rapid-mlx serve --tui`.

The monitor intentionally depends only on existing server endpoints:
`/health` and `/v1/status`. It does not require request metrics middleware.
"""

from __future__ import annotations

import json
import select
import shutil
import sys
import termios
import time
import tty
import urllib.request
from typing import Any

COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "cyan": "\033[36m",
}


def _c(enabled: bool, name: str, text: str) -> str:
    if not enabled:
        return text
    return f"{COLORS.get(name, '')}{text}{COLORS['reset']}"


def _fetch_json(url: str, timeout: float = 2.0) -> tuple[dict[str, Any], str | None]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data if isinstance(data, dict) else {}, None
    except Exception as exc:
        return {}, str(exc)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _fmt_seconds(value: Any) -> str:
    seconds = max(0.0, _num(value))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"
    hours = minutes // 60
    minutes %= 60
    return f"{hours}h{minutes:02d}m"


def _fmt_gb(value: Any) -> str:
    return f"{_num(value):.2f} GB"


def _clamp(text: Any, width: int) -> str:
    if width <= 0:
        return ""
    value = str(text)
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return value[: width - 3] + "..."


def _bar(value: float, limit: float, width: int = 18) -> str:
    if width <= 0:
        return ""
    ratio = 0.0 if limit <= 0 else max(0.0, min(1.0, value / limit))
    filled = int(round(ratio * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _line(width: int, char: str = "-") -> str:
    return char * max(0, width)


def _row(label: str, value: Any, width: int, color: str, tty_on: bool) -> str:
    label_width = min(20, max(11, width // 4))
    value_width = max(0, width - label_width - 1)
    return (
        f"{_c(tty_on, 'dim', label.ljust(label_width))} "
        f"{_c(tty_on, color, _clamp(value, value_width))}"
    )


def _request_tokens(request: dict[str, Any]) -> tuple[int, int]:
    prompt = _integer(request.get("prompt_tokens", request.get("num_prompt_tokens", 0)))
    completion = _integer(
        request.get("completion_tokens", request.get("num_generated_tokens", 0))
    )
    return prompt, completion


def _render_requests(status: dict[str, Any], width: int, tty_on: bool) -> list[str]:
    requests = status.get("requests")
    if not isinstance(requests, list) or not requests:
        return [_c(tty_on, "dim", "No active requests reported by engine.")]

    rows = []
    header = f"{'id':<12} {'state':<10} {'prompt':>7} {'gen':>7} {'tps':>8}"
    rows.append(_c(tty_on, "dim", _clamp(header, width)))
    for item in requests[:8]:
        if not isinstance(item, dict):
            continue
        prompt, completion = _request_tokens(item)
        row = (
            f"{str(item.get('id', item.get('request_id', '-')))[:12]:<12} "
            f"{str(item.get('state', item.get('status', '-')))[:10]:<10} "
            f"{prompt:>7} {completion:>7} {_num(item.get('tokens_per_second')):>8.1f}"
        )
        rows.append(_clamp(row, width))
    return rows


def _build_screen(
    base_url: str,
    pid: int | str,
    interval: float,
    health: dict[str, Any],
    status: dict[str, Any],
    errors: list[str],
    tty_on: bool,
) -> str:
    width, height = shutil.get_terminal_size((100, 32))
    width = max(60, width)
    lines: list[str] = []

    title = "Rapid-MLX live monitor"
    state = str(status.get("status") or health.get("status") or "unknown")
    state_color = "green" if state in {"healthy", "idle"} else "yellow"
    if errors and not health and not status:
        state_color = "red"
    header = f"{title}  pid={pid}  refresh={interval:.1f}s  {base_url}"
    lines.append(_c(tty_on, "bold", _clamp(header, width)))
    lines.append(_line(width))
    lines.append(_row("state", state, width, state_color, tty_on))
    lines.append(
        _row(
            "model",
            status.get("model") or health.get("model_name") or "-",
            width,
            "cyan",
            tty_on,
        )
    )
    lines.append(_row("engine", health.get("engine_type", "-"), width, "cyan", tty_on))
    lines.append(
        _row("uptime", _fmt_seconds(status.get("uptime_s")), width, "green", tty_on)
    )
    lines.append(
        _row(
            "requests",
            f"running={status.get('num_running', 0)} waiting={status.get('num_waiting', 0)} processed={status.get('total_requests_processed', 0)}",
            width,
            "green",
            tty_on,
        )
    )
    lines.append(
        _row(
            "tokens",
            f"prompt={status.get('total_prompt_tokens', 0)} completion={status.get('total_completion_tokens', 0)}",
            width,
            "green",
            tty_on,
        )
    )

    metal = status.get("metal") if isinstance(status.get("metal"), dict) else {}
    lines.append("")
    lines.append(_c(tty_on, "bold", "Metal"))
    active = _num(metal.get("active_memory_gb"))
    peak = _num(metal.get("peak_memory_gb"))
    cache = _num(metal.get("cache_memory_gb"))
    lines.append(
        _row(
            "active",
            f"{_fmt_gb(active)} {_bar(active, max(peak, active, 1.0))}",
            width,
            "yellow",
            tty_on,
        )
    )
    lines.append(_row("peak", _fmt_gb(peak), width, "yellow", tty_on))
    lines.append(_row("cache", _fmt_gb(cache), width, "yellow", tty_on))

    cache_stats = status.get("cache") if isinstance(status.get("cache"), dict) else {}
    if cache_stats:
        lines.append("")
        lines.append(_c(tty_on, "bold", "Cache"))
        hit_rate = _num(cache_stats.get("hit_rate")) * 100
        lines.append(_row("hit rate", f"{hit_rate:.1f}%", width, "green", tty_on))
        lines.append(
            _row(
                "entries",
                cache_stats.get("entry_count", cache_stats.get("num_entries", "-")),
                width,
                "green",
                tty_on,
            )
        )
        lines.append(
            _row(
                "memory",
                f"{cache_stats.get('current_memory_mb', '-')} / {cache_stats.get('max_memory_mb', '-')} MB",
                width,
                "green",
                tty_on,
            )
        )

    lines.append("")
    lines.append(_c(tty_on, "bold", "Active Requests"))
    lines.extend(_render_requests(status, width, tty_on))

    if errors:
        lines.append("")
        lines.append(
            _c(tty_on, "red", "poll errors: " + _clamp(" | ".join(errors), width - 13))
        )

    lines.append("")
    lines.append(_c(tty_on, "dim", "q quits. Ctrl-C quits."))
    return "\n".join(lines[: max(1, height - 1)])


def _read_key() -> str | None:
    if not sys.stdin.isatty():
        return None
    readable, _, _ = select.select([sys.stdin], [], [], 0)
    if not readable:
        return None
    try:
        return sys.stdin.read(1)
    except Exception:
        return None


def run_monitor(base_url: str, interval: float = 1.0, pid: int | str = "?") -> int:
    """Run the full-screen monitor loop until q or Ctrl-C."""
    health_url = base_url.rstrip("/") + "/health"
    status_url = base_url.rstrip("/") + "/v1/status"
    interval = max(0.1, float(interval))
    tty_on = sys.stdout.isatty()

    old_term = None
    if tty_on and sys.stdin.isatty():
        try:
            old_term = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin)
        except Exception:
            old_term = None

    try:
        if tty_on:
            sys.stdout.write("\033[?1049h\033[?25l")
            sys.stdout.flush()

        last_health: dict[str, Any] = {}
        last_status: dict[str, Any] = {}
        while True:
            health, health_error = _fetch_json(health_url)
            status, status_error = _fetch_json(status_url)
            if health:
                last_health = health
            else:
                health = last_health
            if status:
                last_status = status
            else:
                status = last_status

            errors = [e for e in (health_error, status_error) if e]
            screen = _build_screen(
                base_url, pid, interval, health, status, errors, tty_on
            )
            if tty_on:
                sys.stdout.write("\033[H\033[2J")
            sys.stdout.write(screen + "\n")
            sys.stdout.flush()

            deadline = time.time() + interval
            while time.time() < deadline:
                key = _read_key()
                if key in {"q", "Q", "\x03"}:
                    return 0
                time.sleep(0.05)
            if not tty_on:
                sys.stdout.write("\n")
    except KeyboardInterrupt:
        return 0
    finally:
        if old_term is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)
            except Exception:
                pass
        if tty_on:
            sys.stdout.write("\033[?25h\033[?1049l")
            sys.stdout.flush()
    return 0
