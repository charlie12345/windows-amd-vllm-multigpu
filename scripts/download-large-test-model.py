"""Download only the pinned 4-bit Hugging Face shards needed for TP tests."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError


DEFAULT_REPO = "stelterlab/Mistral-Small-24B-Instruct-2501-AWQ"
DEFAULT_REVISION = "cbda099649a0188dd888d44f0e4964d8d982dc9a"
EXPECTED_WEIGHT_BYTES = 14_234_370_648
ALLOW_PATTERNS = (
    "README.md",
    "SYSTEM_PROMPT.txt",
    "config.json",
    "generation_config.json",
    "model-*.safetensors",
    "model.safetensors.index.json",
    "params.json",
    "special_tokens_map.json",
    "tekken.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def cache_root(cache_dir: str | None) -> Path:
    if cache_dir:
        return Path(cache_dir).expanduser().resolve()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser().resolve() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def validate_snapshot(model_path: Path) -> None:
    shards = sorted(model_path.glob("model-*.safetensors"))
    required = (model_path / "config.json", model_path / "model.safetensors.index.json")
    if len(shards) != 3 or not all(path.is_file() for path in required):
        raise RuntimeError(f"Incomplete model snapshot at {model_path}")
    shard_bytes = sum(path.stat().st_size for path in shards)
    if shard_bytes != EXPECTED_WEIGHT_BYTES:
        raise RuntimeError(
            f"Expected {EXPECTED_WEIGHT_BYTES} weight bytes, found {shard_bytes}"
        )
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    quant = config.get("quantization_config", {})
    expected_quant = {
        "quant_method": "awq",
        "bits": 4,
        "group_size": 128,
        "zero_point": True,
        "version": "gemm",
    }
    mismatches = {
        key: (quant.get(key), expected)
        for key, expected in expected_quant.items()
        if quant.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Unexpected AWQ configuration: {mismatches}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    root = cache_root(args.cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    try:
        existing_path = Path(
            snapshot_download(
                repo_id=args.repo,
                revision=args.revision,
                cache_dir=str(root),
                allow_patterns=ALLOW_PATTERNS,
                ignore_patterns=("consolidated.safetensors",),
                local_files_only=True,
            )
        )
        validate_snapshot(existing_path)
        print(f"WAVMG_MODEL_PATH={existing_path.resolve()}")
        return 0
    except LocalEntryNotFoundError:
        if args.local_files_only:
            raise

    if not args.local_files_only:
        free_bytes = shutil.disk_usage(root).free
        reserve_bytes = 5 * 1024**3
        if free_bytes < EXPECTED_WEIGHT_BYTES + reserve_bytes:
            raise RuntimeError(
                "The pinned model needs about 13.3 GiB plus cache overhead; "
                f"only {free_bytes / 1024**3:.1f} GiB is free at {root}."
            )
        info = HfApi().model_info(args.repo, revision=args.revision)
        if info.sha != args.revision:
            raise RuntimeError(
                f"Hugging Face resolved {args.revision} to unexpected {info.sha}"
            )
        license_name = (info.card_data or {}).get("license")
        print(
            f"Downloading {args.repo}@{args.revision} "
            f"(license={license_name}, weights=13.3 GiB, format=AWQ W4A16 G128)"
        )

    model_path = Path(
        snapshot_download(
            repo_id=args.repo,
            revision=args.revision,
            cache_dir=str(root),
            allow_patterns=ALLOW_PATTERNS,
            ignore_patterns=("consolidated.safetensors",),
            local_files_only=args.local_files_only,
            max_workers=args.max_workers,
        )
    )
    validate_snapshot(model_path)
    print(f"WAVMG_MODEL_PATH={model_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
