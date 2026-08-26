#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Carlo Pasquale and project contributors
"""Build a standalone native vLLM MTP drafter for the local Qwen3.8 AWQ model."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download


SOURCE_REPO = "Qwen/Qwen3.8-27B"
SOURCE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SOURCE_SHARD = "model-00018-of-00018.safetensors"
SOURCE_SHARD_BYTES = 3_392_197_344
SOURCE_SHARD_SHA256 = (
    "1d3479509e21494658f9b64d317f5ea8e55c4025d28c702d6c4d0b356ce8ea06"
)
AWQ_WEIGHT = "model.safetensors"
DRAFT_WEIGHT = "model.safetensors"
SHARED_KEYS = ("model.embed_tokens.weight", "lm_head.weight")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as stream:
        header_size_raw = stream.read(8)
        if len(header_size_raw) != 8:
            raise ValueError(f"Invalid safetensors header: {path}")
        header_size = struct.unpack("<Q", header_size_raw)[0]
        header = json.loads(stream.read(header_size).decode("utf-8"))
    return header, 8 + header_size


def copy_exact(source, destination, length: int) -> None:
    remaining = length
    while remaining:
        chunk = source.read(min(16 * 1024 * 1024, remaining))
        if not chunk:
            raise EOFError("Unexpected end of safetensors data")
        destination.write(chunk)
        remaining -= len(chunk)


def build_draft_safetensors(
    output: Path,
    sources: list[tuple[Path, list[str]]],
) -> dict[str, int]:
    entries: list[tuple[Path, int, int, str, dict]] = []
    output_header: dict[str, object] = {
        "__metadata__": {
            "format": "pt",
            "source_repo": SOURCE_REPO,
            "source_revision": SOURCE_REVISION,
            "source_shard": SOURCE_SHARD,
            "purpose": "standalone Qwen3.8 native MTP drafter for vLLM",
        }
    }
    output_offset = 0
    sizes: dict[str, int] = {}
    for source_path, keys in sources:
        source_header, source_data_start = read_safetensors_header(source_path)
        for key in keys:
            if key not in source_header:
                raise ValueError(f"{source_path} is missing tensor {key}")
            source_entry = source_header[key]
            source_start, source_end = source_entry["data_offsets"]
            length = source_end - source_start
            output_header[key] = {
                "dtype": source_entry["dtype"],
                "shape": source_entry["shape"],
                "data_offsets": [output_offset, output_offset + length],
            }
            entries.append(
                (
                    source_path,
                    source_data_start + source_start,
                    length,
                    key,
                    source_entry,
                )
            )
            sizes[key] = length
            output_offset += length

    header_bytes = json.dumps(output_header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * (-len(header_bytes) % 8)
    with output.open("wb") as destination:
        destination.write(struct.pack("<Q", len(header_bytes)))
        destination.write(header_bytes)
        for source_path, source_offset, length, _, _ in entries:
            with source_path.open("rb") as source:
                source.seek(source_offset)
                copy_exact(source, destination, length)
    return sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--awq-dir",
        type=Path,
        default=Path(r"G:\AI-models\Qwen3.8-27B-AWQ-INT4"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"G:\AI-models\Qwen3.8-27B-MTP-vLLM"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.awq_dir.resolve()
    output = args.output_dir.resolve()

    source_weight = source / AWQ_WEIGHT
    source_config = source / "config.json"
    if not source_weight.is_file() or not source_config.is_file():
        raise FileNotFoundError(f"Incomplete AWQ checkpoint: {source}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output}. "
            "Remove that generated drafter explicitly before rebuilding it."
        )
    output.mkdir(parents=True, exist_ok=True)

    for item in source.iterdir():
        if item.is_file() and item.name not in (AWQ_WEIGHT, "config.json"):
            shutil.copy2(item, output / item.name)

    with source_config.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    config.pop("quantization_config", None)
    config["architectures"] = ["Qwen3_5MTP"]
    with (output / "config.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(config, stream, indent=2, sort_keys=False)
        stream.write("\n")

    with tempfile.TemporaryDirectory(prefix="qwen38-mtp-source-", dir=output.parent) as tmp:
        downloaded = Path(
            hf_hub_download(
                repo_id=SOURCE_REPO,
                filename=SOURCE_SHARD,
                revision=SOURCE_REVISION,
                local_dir=tmp,
            )
        )
        if downloaded.stat().st_size != SOURCE_SHARD_BYTES:
            raise ValueError(
                f"Unexpected source shard size: {downloaded.stat().st_size}"
            )
        actual_sha256 = sha256(downloaded)
        if actual_sha256 != SOURCE_SHARD_SHA256:
            raise ValueError(f"Unexpected source shard SHA-256: {actual_sha256}")

        source_header, _ = read_safetensors_header(downloaded)
        mtp_keys = sorted(key for key in source_header if key.startswith("mtp."))
        if not mtp_keys:
            raise ValueError("The pinned source shard contains no MTP tensors.")
        sizes = build_draft_safetensors(
            output / DRAFT_WEIGHT,
            [(source_weight, list(SHARED_KEYS)), (downloaded, mtp_keys)],
        )

    mtp_bytes = sum(sizes[key] for key in mtp_keys)
    shared_bytes = sum(sizes[key] for key in SHARED_KEYS)

    manifest = {
        "target_awq_directory": str(source),
        "mtp_source_repo": SOURCE_REPO,
        "mtp_source_revision": SOURCE_REVISION,
        "mtp_source_shard": SOURCE_SHARD,
        "mtp_source_shard_bytes": SOURCE_SHARD_BYTES,
        "mtp_source_shard_sha256": SOURCE_SHARD_SHA256,
        "mtp_tensor_count": len(mtp_keys),
        "mtp_tensor_bytes": mtp_bytes,
        "shared_tensor_names": list(SHARED_KEYS),
        "shared_tensor_bytes": shared_bytes,
        "draft_weight_file": DRAFT_WEIGHT,
        "draft_weight_bytes": (output / DRAFT_WEIGHT).stat().st_size,
        "draft_quantization": "BF16/unquantized",
        "target_quantization": "compressed-tensors W4A16 INT4 group-128",
        "license": "Apache-2.0; retain Qwen and AWQ checkpoint attribution",
    }
    with (output / "mtp-draft.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")

    notice = f"""# Qwen3.8-27B native MTP drafter for vLLM

This local-only checkpoint contains the BF16 embedding and LM head from
`{source}` plus the official native MTP tensors extracted from `{SOURCE_REPO}`
at revision `{SOURCE_REVISION}`. Use it as the MTP speculative model alongside
the separate compressed-tensors W4A16 target; it is not a standalone target
language model.

Both sources declare Apache-2.0. Retain the original Qwen and quantizer
attribution. This generated directory and all model weights stay outside Git.
"""
    (output / "MTP-DRAFT.md").write_text(notice, encoding="utf-8", newline="\n")

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
