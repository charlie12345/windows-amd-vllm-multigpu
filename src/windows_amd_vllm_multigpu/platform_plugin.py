"""Lazy vLLM platform-plugin entry point."""

from __future__ import annotations

import os
import sys


def windows_amd_platform_plugin() -> str | None:
    """Activate only when the user explicitly enables this Windows AMD plugin."""
    enabled = os.environ.get("WAVMG_ENABLE", "").lower()
    if enabled not in {"1", "true", "yes", "on"} or sys.platform != "win32":
        return None
    try:
        import torch
    except ImportError:
        return None
    if torch.version.hip is None:
        return None
    return "windows_amd_vllm_multigpu.platform.WindowsAmdMultiGpuPlatform"
