"""Report the exact distributed and GPU capabilities of the active wheel."""

from __future__ import annotations

import json
import platform
import sys

import torch


def main() -> int:
    dist = torch.distributed
    device_count = torch.cuda.device_count()
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_file": torch.__file__,
        "hip": torch.version.hip,
        "distributed_available": dist.is_available(),
        "gloo_available": dist.is_gloo_available() if dist.is_available() else False,
        "nccl_available": dist.is_nccl_available() if dist.is_available() else False,
        "device_count": device_count,
        "devices": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "arch": torch.cuda.get_device_properties(index).gcnArchName,
            }
            for index in range(device_count)
        ],
        "peer_access": {
            f"{source}->{target}": torch.cuda.can_device_access_peer(source, target)
            for source in range(device_count)
            for target in range(device_count)
            if source != target
        },
        "build_config": torch.__config__.show(),
    }
    print(json.dumps(result, indent=2))

    required = (
        result["distributed_available"]
        and result["gloo_available"]
        and device_count >= 2
    )
    return 0 if required else 2


if __name__ == "__main__":
    raise SystemExit(main())

