"""Generate deterministic token IDs with TP=1 or TP=2 for parity checks."""

from __future__ import annotations

import argparse
import json
import os
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, choices=(1, 2), required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--use-rccl", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--use-d3d12", action=argparse.BooleanOptionalAction, default=None
    )
    args = parser.parse_args()

    os.environ.update(
        {
            "HIP_VISIBLE_DEVICES": "0,1",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_USE_V2_MODEL_RUNNER": "0",
            "VLLM_DISTRIBUTED_USE_SPLIT_GROUP": "0",
            "WAVMG_ENABLE": "1",
            "VLLM_PLUGINS": "windows_amd_multigpu",
            "HF_HUB_OFFLINE": "1",
        }
    )
    if args.use_rccl is not None:
        os.environ["WAVMG_USE_RCCL"] = "1" if args.use_rccl else "0"
    if args.use_d3d12 is not None:
        os.environ["WAVMG_USE_D3D12"] = "1" if args.use_d3d12 else "0"

    from vllm import LLM, SamplingParams

    prompts = [
        "The capital of France is",
        "Two plus two equals",
    ]
    load_started = time.perf_counter()
    llm_options = {}
    if args.max_num_batched_tokens is not None:
        llm_options["max_num_batched_tokens"] = args.max_num_batched_tokens
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        distributed_executor_backend="mp",
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        compilation_config=0,
        enable_prefix_caching=False,
        **llm_options,
    )
    load_seconds = time.perf_counter() - load_started
    try:
        generation_started = time.perf_counter()
        outputs = llm.generate(
            prompts,
            SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=0),
            use_tqdm=False,
        )
        generation_seconds = time.perf_counter() - generation_started
        result = {
            "tensor_parallel_size": args.tp,
            "model": args.model,
            "load_seconds": load_seconds,
            "generation_seconds": generation_seconds,
            "use_rccl": os.environ.get("WAVMG_USE_RCCL") == "1",
            "use_d3d12": os.environ.get("WAVMG_USE_D3D12") == "1",
            "outputs": [
                {
                    "prompt": output.prompt,
                    "token_ids": list(output.outputs[0].token_ids),
                    "text": output.outputs[0].text,
                }
                for output in outputs
            ],
        }
        print("WAVMG_RESULT=" + json.dumps(result, ensure_ascii=False))
    finally:
        # Do not rely on interpreter finalization to reap spawned GPU workers.
        # This is intentionally defensive because a loaded Windows DLL cannot
        # be replaced until every worker has exited.
        engine_core = getattr(llm.llm_engine, "engine_core", None)
        shutdown = getattr(engine_core, "shutdown", None)
        if callable(shutdown):
            shutdown(timeout=30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
