"""ROCm runtime discovery shared by the native Windows transports."""

from __future__ import annotations

import os
from pathlib import Path


def rocm_bin_directories() -> list[Path]:
    import torch

    site_packages = Path(torch.__file__).resolve().parent.parent
    configured = os.environ.get("WAVMG_ROCM_BIN")
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser().resolve())
    candidates.extend(
        (
            site_packages / "_rocm_sdk_devel" / "bin",
            site_packages / "_rocm_sdk_core" / "bin",
            site_packages / "_rocm_sdk_libraries_custom" / "bin",
            Path(torch.__file__).resolve().parent / "lib",
        )
    )
    return list(dict.fromkeys(path for path in candidates if path.is_dir()))


def find_hip_runtime() -> Path:
    configured = os.environ.get("WAVMG_HIP_RUNTIME_DLL")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"WAVMG_HIP_RUNTIME_DLL does not exist: {candidate}")

    for directory in rocm_bin_directories():
        candidate = directory / "amdhip64_7.dll"
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "amdhip64_7.dll was not found beside the installed PyTorch ROCm packages; "
        "set WAVMG_HIP_RUNTIME_DLL or WAVMG_ROCM_BIN"
    )
