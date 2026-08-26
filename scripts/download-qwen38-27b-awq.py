"""Download and validate the pinned Qwen3.8-27B W4A16 checkpoint.

The checkpoint is native SafeTensors using the compressed-tensors
pack-quantized W4A16 schema.  It is intentionally not GGUF or NVFP4.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


DEFAULT_REPO = "abihsoro/Qwen3.8-27B-AWQ-INT4"
DEFAULT_REVISION = "f2e0cac39907e7b1ed7fdb210363dd33cc18f993"
DEFAULT_LOCAL_DIR = Path("G:/AI-models/Qwen3.8-27B-AWQ-INT4")
EXPECTED_WEIGHT_BYTES = 17_646_863_912
ALLOW_PATTERNS = (
    ".gitattributes",
    "README.md",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "recipe.yaml",
    "tokenizer.json",
    "tokenizer_config.json",
)


def validate_snapshot(model_path: Path) -> None:
    required = (
        model_path / "config.json",
        model_path / "model.safetensors",
        model_path / "tokenizer.json",
    )
    if not all(path.is_file() for path in required):
        missing = [str(path) for path in required if not path.is_file()]
        raise RuntimeError(f"Incomplete model snapshot; missing: {missing}")

    weight_path = model_path / "model.safetensors"
    if weight_path.stat().st_size != EXPECTED_WEIGHT_BYTES:
        raise RuntimeError(
            f"Expected {EXPECTED_WEIGHT_BYTES} weight bytes, found "
            f"{weight_path.stat().st_size}"
        )

    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    quant = config.get("quantization_config", {})
    groups = quant.get("config_groups", {})
    group = groups.get("group_0", {})
    weights = group.get("weights", {})
    expected = {
        "architecture": "Qwen3_5ForCausalLM",
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "weight_bits": 4,
        "group_size": 128,
        "symmetric": True,
        "weight_type": "int",
        "activation_quantization": None,
    }
    actual = {
        "architecture": (config.get("architectures") or [None])[0],
        "quant_method": quant.get("quant_method"),
        "format": quant.get("format"),
        "weight_bits": weights.get("num_bits"),
        "group_size": weights.get("group_size"),
        "symmetric": weights.get("symmetric"),
        "weight_type": weights.get("type"),
        "activation_quantization": group.get("input_activations"),
    }
    mismatches = {
        key: {"actual": actual[key], "expected": value}
        for key, value in expected.items()
        if actual[key] != value
    }
    if mismatches:
        raise RuntimeError(f"Unexpected W4A16 configuration: {mismatches}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    model_path = args.local_dir.expanduser().resolve()
    if model_path.exists():
        try:
            validate_snapshot(model_path)
            print(f"WAVMG_MODEL_PATH={model_path}")
            return 0
        except RuntimeError:
            if args.local_files_only:
                raise
    elif args.local_files_only:
        raise RuntimeError(f"No local checkpoint exists at {model_path}")

    info = HfApi().model_info(
        args.repo,
        revision=args.revision,
        files_metadata=True,
    )
    if info.sha != args.revision:
        raise RuntimeError(f"Revision resolved to unexpected commit {info.sha}")
    license_name = (
        (info.card_data or {}).get("license")
        if isinstance(info.card_data, dict)
        else getattr(info.card_data, "license", None)
    )
    if license_name != "apache-2.0":
        raise RuntimeError(f"Unexpected model license: {license_name!r}")
    remote_weights = {
        sibling.rfilename: sibling.size for sibling in info.siblings
    }.get("model.safetensors")
    if remote_weights != EXPECTED_WEIGHT_BYTES:
        raise RuntimeError(
            f"Remote model.safetensors is {remote_weights} bytes; expected "
            f"{EXPECTED_WEIGHT_BYTES}"
        )

    model_path.mkdir(parents=True, exist_ok=True)
    existing_weight_bytes = sum(
        path.stat().st_size for path in model_path.glob("*.safetensors")
    )
    bytes_left = max(0, EXPECTED_WEIGHT_BYTES - existing_weight_bytes)
    reserve_bytes = 5 * 1024**3
    free_bytes = shutil.disk_usage(model_path).free
    if free_bytes < bytes_left + reserve_bytes:
        raise RuntimeError(
            f"Download needs about {bytes_left / 1024**3:.1f} GiB plus a "
            f"5 GiB reserve; only {free_bytes / 1024**3:.1f} GiB is free."
        )

    print(
        f"Downloading {args.repo}@{args.revision} to {model_path} "
        "(Apache-2.0, SafeTensors compressed-tensors W4A16 INT4 G128)"
    )
    snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        local_dir=model_path,
        allow_patterns=ALLOW_PATTERNS,
        local_files_only=False,
        max_workers=args.max_workers,
    )
    validate_snapshot(model_path)
    print(f"WAVMG_MODEL_PATH={model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
