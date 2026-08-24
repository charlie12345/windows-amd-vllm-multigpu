"""Opt-in vLLM ROCm platform backed by the Windows AMD communicator."""

from __future__ import annotations

from vllm.platforms.rocm import RocmPlatform


class WindowsAmdMultiGpuPlatform(RocmPlatform):
    """ROCm platform variant that uses CPU Gloo only for rendezvous/control."""

    dist_backend = "gloo"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return (
            "windows_amd_vllm_multigpu.vllm_communicator."
            "WindowsAmdMultiGpuCommunicator"
        )

    @classmethod
    def use_custom_allreduce(cls) -> bool:
        return False

    @classmethod
    def use_custom_op_collectives(cls) -> bool:
        # Keep collectives as ordinary Python calls. The mapped-memory transport
        # is stream ordered, but is not currently safe for graph capture.
        return False

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        return False

    @classmethod
    def check_and_update_config(cls, vllm_config) -> None:
        super().check_and_update_config(vllm_config)
        parallel = vllm_config.parallel_config
        tp = int(parallel.tensor_parallel_size)
        pp = int(parallel.pipeline_parallel_size)
        if tp not in (1, 2):
            raise ValueError("Windows AMD transport currently supports TP=1 or TP=2")
        if pp != 1:
            raise ValueError(
                "pipeline parallelism needs the version-pinned vLLM broadcast patch"
            )
        unsupported = {
            "data_parallel_size": getattr(parallel, "data_parallel_size", 1),
            "decode_context_parallel_size": getattr(
                parallel, "decode_context_parallel_size", 1
            ),
            "prefill_context_parallel_size": getattr(
                parallel, "prefill_context_parallel_size", 1
            ),
        }
        enabled = {name: value for name, value in unsupported.items() if value != 1}
        if enabled:
            raise ValueError(f"unsupported Windows AMD parallel modes: {enabled}")
        if getattr(parallel, "enable_expert_parallel", False):
            raise ValueError("expert parallel/all-to-all is not implemented")
