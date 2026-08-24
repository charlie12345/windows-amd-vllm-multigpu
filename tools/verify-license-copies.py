"""Verify packaged upstream license texts against the pinned sandboxes."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAIRS = (
    (
        "RCCL license",
        PROJECT_ROOT / "sandbox/rocm-systems/projects/rccl/LICENSE.txt",
        PROJECT_ROOT / "LICENSES/RCCL-UPSTREAM-LICENSE.txt",
    ),
    (
        "RCCL notices",
        PROJECT_ROOT / "sandbox/rocm-systems/projects/rccl/NOTICES.txt",
        PROJECT_ROOT / "LICENSES/RCCL-UPSTREAM-NOTICES.txt",
    ),
    (
        "RCCL third-party notices",
        PROJECT_ROOT / "sandbox/rocm-systems/projects/rccl/ThirdPartyNotices.txt",
        PROJECT_ROOT / "LICENSES/RCCL-UPSTREAM-ThirdPartyNotices.txt",
    ),
    (
        "vLLM license",
        PROJECT_ROOT / "sandbox/vllm/LICENSE",
        PROJECT_ROOT / "LICENSES/VLLM-UPSTREAM-LICENSE.txt",
    ),
    (
        "vLLM notice",
        PROJECT_ROOT / "sandbox/vllm/NOTICE",
        PROJECT_ROOT / "LICENSES/VLLM-UPSTREAM-NOTICE.txt",
    ),
)


def normalized(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").rstrip("\n")


def main() -> int:
    for label, upstream, packaged in PAIRS:
        if not upstream.is_file():
            raise FileNotFoundError(
                f"Missing pinned upstream file for {label}: {upstream}"
            )
        if normalized(upstream) != normalized(packaged):
            raise RuntimeError(f"Packaged {label} does not match {upstream}")
        print(f"PASS {label}: {packaged.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
