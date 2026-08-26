"""Download and validate the pinned official Qwen3.8-27B DFlash2 drafter."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


DEFAULT_REPO = "z-lab/Qwen3.8-27B-DFlash2"
DEFAULT_REVISION = "50307d4c4cde6860d4eee73e2547cd786fe8e8a4"
DEFAULT_LOCAL_DIR = Path("G:/AI-models/Qwen3.8-27B-DFlash2")
EXPECTED_WEIGHT_BYTES = 3_848_817_896
ALLOW_PATTERNS = (
    ".gitattributes",
    "README.md",
    "config.json",
    "model.safetensors",
)


def validate_snapshot(model_path: Path) -> None:
    config_path = model_path / "config.json"
    weight_path = model_path / "model.safetensors"
    missing = [str(path) for path in (config_path, weight_path) if not path.is_file()]
    if missing:
        raise RuntimeError(f"Incomplete DFlash2 snapshot; missing: {missing}")
    if weight_path.stat().st_size != EXPECTED_WEIGHT_BYTES:
        raise RuntimeError(
            f"Expected {EXPECTED_WEIGHT_BYTES} weight bytes, found "
            f"{weight_path.stat().st_size}"
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    dflash = config.get("dflash_config", {})
    expected = {
        "architecture": "DFlash2DraftModel",
        "block_size": 8,
        "num_speculative_tokens": 7,
        "selector_top_k": 16,
    }
    actual = {
        "architecture": (config.get("architectures") or [None])[0],
        "block_size": dflash.get("block_size"),
        "num_speculative_tokens": (
            dflash.get("block_size") - 1
            if isinstance(dflash.get("block_size"), int)
            else None
        ),
        "selector_top_k": dflash.get("selector_top_k"),
    }
    mismatches = {
        key: {"actual": actual[key], "expected": value}
        for key, value in expected.items()
        if actual[key] != value
    }
    if mismatches:
        raise RuntimeError(f"Unexpected DFlash2 configuration: {mismatches}")


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
            print(f"WAVMG_DFLASH2_MODEL_PATH={model_path}")
            return 0
        except RuntimeError:
            if args.local_files_only:
                raise
    elif args.local_files_only:
        raise RuntimeError(f"No local DFlash2 checkpoint exists at {model_path}")

    info = HfApi().model_info(args.repo, revision=args.revision, files_metadata=True)
    if info.sha != args.revision:
        raise RuntimeError(f"Revision resolved to unexpected commit {info.sha}")
    license_name = (
        (info.card_data or {}).get("license")
        if isinstance(info.card_data, dict)
        else getattr(info.card_data, "license", None)
    )
    if license_name != "apache-2.0":
        raise RuntimeError(f"Unexpected model license: {license_name!r}")
    remote_weight_bytes = {
        sibling.rfilename: sibling.size for sibling in info.siblings
    }.get("model.safetensors")
    if remote_weight_bytes != EXPECTED_WEIGHT_BYTES:
        raise RuntimeError(
            f"Remote model.safetensors is {remote_weight_bytes} bytes; expected "
            f"{EXPECTED_WEIGHT_BYTES}"
        )

    model_path.mkdir(parents=True, exist_ok=True)
    bytes_left = max(
        0,
        EXPECTED_WEIGHT_BYTES
        - sum(path.stat().st_size for path in model_path.glob("*.safetensors")),
    )
    reserve_bytes = 3 * 1024**3
    free_bytes = shutil.disk_usage(model_path).free
    if free_bytes < bytes_left + reserve_bytes:
        raise RuntimeError(
            f"Download needs about {bytes_left / 1024**3:.1f} GiB plus a "
            f"3 GiB reserve; only {free_bytes / 1024**3:.1f} GiB is free."
        )

    print(
        f"Downloading {args.repo}@{args.revision} to {model_path} "
        "(Apache-2.0, official SafeTensors DFlash2 drafter)"
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
    print(f"WAVMG_DFLASH2_MODEL_PATH={model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
