# SPDX-License-Identifier: Apache-2.0
"""Apple Silicon hardware profiles.

Each profile captures the specs that matter for LLM inference:
- memory_gb: unified memory (determines max model + KV cache)
- bandwidth_gbs: memory bandwidth (determines decode tok/s)
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareProfile:
    id: str
    name: str
    chip: str
    memory_gb: int
    bandwidth_gbs: int  # GB/s
    generation: str  # m1, m2, m3, m4

    @property
    def usable_memory_gb(self) -> float:
        """Memory available for model + KV cache (minus OS overhead)."""
        return self.memory_gb - 4  # ~4GB for macOS + apps


# fmt: off
HARDWARE_PROFILES: dict[str, HardwareProfile] = {
    # M1 family
    "m1-8":           HardwareProfile("m1-8",           "M1 8GB",           "M1",          8,   68, "m1"),
    "m1-16":          HardwareProfile("m1-16",          "M1 16GB",          "M1",         16,   68, "m1"),
    "m1-pro-16":      HardwareProfile("m1-pro-16",      "M1 Pro 16GB",      "M1 Pro",     16,  200, "m1"),
    "m1-pro-32":      HardwareProfile("m1-pro-32",      "M1 Pro 32GB",      "M1 Pro",     32,  200, "m1"),
    "m1-max-32":      HardwareProfile("m1-max-32",      "M1 Max 32GB",      "M1 Max",     32,  400, "m1"),
    "m1-max-64":      HardwareProfile("m1-max-64",      "M1 Max 64GB",      "M1 Max",     64,  400, "m1"),
    "m1-ultra-64":    HardwareProfile("m1-ultra-64",    "M1 Ultra 64GB",    "M1 Ultra",   64,  800, "m1"),
    "m1-ultra-128":   HardwareProfile("m1-ultra-128",   "M1 Ultra 128GB",   "M1 Ultra",  128,  800, "m1"),
    # M2 family
    "m2-8":           HardwareProfile("m2-8",           "M2 8GB",           "M2",          8,  100, "m2"),
    "m2-16":          HardwareProfile("m2-16",          "M2 16GB",          "M2",         16,  100, "m2"),
    "m2-24":          HardwareProfile("m2-24",          "M2 24GB",          "M2",         24,  100, "m2"),
    "m2-pro-16":      HardwareProfile("m2-pro-16",      "M2 Pro 16GB",      "M2 Pro",     16,  200, "m2"),
    "m2-pro-32":      HardwareProfile("m2-pro-32",      "M2 Pro 32GB",      "M2 Pro",     32,  200, "m2"),
    "m2-max-32":      HardwareProfile("m2-max-32",      "M2 Max 32GB",      "M2 Max",     32,  400, "m2"),
    "m2-max-64":      HardwareProfile("m2-max-64",      "M2 Max 64GB",      "M2 Max",     64,  400, "m2"),
    "m2-max-96":      HardwareProfile("m2-max-96",      "M2 Max 96GB",      "M2 Max",     96,  400, "m2"),
    "m2-ultra-64":    HardwareProfile("m2-ultra-64",    "M2 Ultra 64GB",    "M2 Ultra",   64,  800, "m2"),
    "m2-ultra-128":   HardwareProfile("m2-ultra-128",   "M2 Ultra 128GB",   "M2 Ultra",  128,  800, "m2"),
    "m2-ultra-192":   HardwareProfile("m2-ultra-192",   "M2 Ultra 192GB",   "M2 Ultra",  192,  800, "m2"),
    # M3 family
    "m3-8":           HardwareProfile("m3-8",           "M3 8GB",           "M3",          8,  100, "m3"),
    "m3-16":          HardwareProfile("m3-16",          "M3 16GB",          "M3",         16,  100, "m3"),
    "m3-24":          HardwareProfile("m3-24",          "M3 24GB",          "M3",         24,  100, "m3"),
    "m3-pro-18":      HardwareProfile("m3-pro-18",      "M3 Pro 18GB",      "M3 Pro",     18,  150, "m3"),
    "m3-pro-36":      HardwareProfile("m3-pro-36",      "M3 Pro 36GB",      "M3 Pro",     36,  150, "m3"),
    "m3-max-36":      HardwareProfile("m3-max-36",      "M3 Max 36GB",      "M3 Max",     36,  400, "m3"),
    "m3-max-48":      HardwareProfile("m3-max-48",      "M3 Max 48GB",      "M3 Max",     48,  400, "m3"),
    "m3-max-64":      HardwareProfile("m3-max-64",      "M3 Max 64GB",      "M3 Max",     64,  400, "m3"),
    "m3-max-96":      HardwareProfile("m3-max-96",      "M3 Max 96GB",      "M3 Max",     96,  400, "m3"),
    "m3-max-128":     HardwareProfile("m3-max-128",     "M3 Max 128GB",     "M3 Max",    128,  400, "m3"),
    "m3-ultra-192":   HardwareProfile("m3-ultra-192",   "M3 Ultra 192GB",   "M3 Ultra",  192,  800, "m3"),
    "m3-ultra-256":   HardwareProfile("m3-ultra-256",   "M3 Ultra 256GB",   "M3 Ultra",  256,  800, "m3"),
    # M4 family
    "m4-16":          HardwareProfile("m4-16",          "M4 16GB",          "M4",         16,  120, "m4"),
    "m4-24":          HardwareProfile("m4-24",          "M4 24GB",          "M4",         24,  120, "m4"),
    "m4-32":          HardwareProfile("m4-32",          "M4 32GB",          "M4",         32,  120, "m4"),
    "m4-pro-24":      HardwareProfile("m4-pro-24",      "M4 Pro 24GB",      "M4 Pro",     24,  273, "m4"),
    "m4-pro-48":      HardwareProfile("m4-pro-48",      "M4 Pro 48GB",      "M4 Pro",     48,  273, "m4"),
    "m4-max-36":      HardwareProfile("m4-max-36",      "M4 Max 36GB",      "M4 Max",     36,  546, "m4"),
    "m4-max-48":      HardwareProfile("m4-max-48",      "M4 Max 48GB",      "M4 Max",     48,  546, "m4"),
    "m4-max-64":      HardwareProfile("m4-max-64",      "M4 Max 64GB",      "M4 Max",     64,  546, "m4"),
    "m4-max-128":     HardwareProfile("m4-max-128",     "M4 Max 128GB",     "M4 Max",    128,  546, "m4"),
}
# fmt: on


def detect_hardware() -> HardwareProfile | None:
    """Auto-detect the current Mac's hardware profile.

    Returns None on non-macOS or if detection fails.
    """
    if platform.system() != "Darwin":
        return None

    try:
        chip = (
            subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        mem_bytes = int(
            subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, ValueError):
        return None

    mem_gb = mem_bytes // (1024**3)

    # Normalize chip name: "Apple M3 Ultra" → "m3-ultra"
    chip_lower = chip.lower().replace("apple ", "")
    parts = chip_lower.split()  # ["m3", "ultra"] or ["m4", "max"] or ["m3"]
    if not parts:
        return None

    if len(parts) >= 2:
        chip_key = f"{parts[0]}-{parts[1]}-{mem_gb}"
    else:
        chip_key = f"{parts[0]}-{mem_gb}"

    if chip_key in HARDWARE_PROFILES:
        return HARDWARE_PROFILES[chip_key]

    # Fuzzy match: find closest memory tier for same chip
    prefix = "-".join(parts)
    candidates = [
        hp for hp in HARDWARE_PROFILES.values() if hp.id.startswith(prefix + "-")
    ]
    if candidates:
        # Pick the profile with closest memory_gb <= actual
        valid = [c for c in candidates if c.memory_gb <= mem_gb]
        if valid:
            return max(valid, key=lambda c: c.memory_gb)
        return min(candidates, key=lambda c: c.memory_gb)

    return None
