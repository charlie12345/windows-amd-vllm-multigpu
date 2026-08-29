"""Validate fail-closed Windows AMD parallel-configuration gates."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

from vllm.platforms.rocm import RocmPlatform

from windows_amd_vllm_multigpu.platform import WindowsAmdMultiGpuPlatform


def _config(*, tp: int = 2, all2all: bool = False) -> SimpleNamespace:
    parallel = SimpleNamespace(
        tensor_parallel_size=tp,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        enable_expert_parallel=True,
        use_all2all=all2all,
    )
    return SimpleNamespace(parallel_config=parallel)


def _expect_rejection(config: SimpleNamespace, text: str) -> None:
    try:
        WindowsAmdMultiGpuPlatform.check_and_update_config(config)
    except ValueError as error:
        if text not in str(error):
            raise AssertionError((text, str(error))) from error
    else:
        raise AssertionError(f"expected rejection containing {text!r}")


def main() -> int:
    with patch.object(
        RocmPlatform,
        "check_and_update_config",
        new=classmethod(lambda cls, config: None),
    ):
        os.environ.pop("WAVMG_ALLOW_TP_EXPERT_PARALLEL", None)
        _expect_rejection(_config(), "experimental")

        os.environ["WAVMG_ALLOW_TP_EXPERT_PARALLEL"] = "1"
        WindowsAmdMultiGpuPlatform.check_and_update_config(_config())
        _expect_rejection(_config(all2all=True), "all-to-all")
        _expect_rejection(_config(tp=1), "experimental")

    print("WAVMG_PLATFORM_CONFIG=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
