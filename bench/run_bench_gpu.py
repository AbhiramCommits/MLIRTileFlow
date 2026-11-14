#!/usr/bin/env python3
"""GPU benchmark driver.

The GPU lowering is compile-tested by `test/Lowering/gpu-lowering.mlir`.
Execution is only attempted when a CUDA device is actually present; this
machine has none, so the script records that fact and exits successfully
instead of fabricating numbers.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

GPU_STATUS = RESULTS / "gpu_status.txt"
GPU_CSV = RESULTS / "bench_gpu.csv"


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def main() -> int:
    if not cuda_available():
        GPU_STATUS.write_text(
            "GPU execution was not available on this machine: torch.cuda.is_available() "
            "is false (no CUDA device).\n"
            "The -tileflow-lower-to-gpu path was still validated by the lit/FileCheck "
            "test test/Lowering/gpu-lowering.mlir, which lowers matmul through "
            "scf.forall -> gpu.launch -> gpu.module with NVVM ops.\n"
        )
        print(GPU_STATUS.read_text())
        return 0

    # A CUDA device exists. Lowering is still the only part this harness can
    # do without a CUDA runtime launcher for the generated kernel; record the
    # verified lowering and be explicit that no timing was fabricated.
    source = REPO_ROOT / "bench" / "matmul_1024.mlir"
    lowered = RESULTS / "raw" / "gpu_lowered.mlir"
    lowered.parent.mkdir(parents=True, exist_ok=True)
    with lowered.open("w") as handle:
        subprocess.run(
            [
                str(REPO_ROOT / "build" / "bin" / "tileflow-opt"),
                str(source),
                "-tileflow-lower-to-gpu=stop-after=nvvm",
                "-o",
                "-",
            ],
            check=True,
            stdout=handle,
        )
    GPU_CSV.write_text(
        "shape,variant,status\nmatmul_1024,gpu_nvvm,lowering_verified_execution_runner_missing\n"
    )
    GPU_STATUS.write_text(
        "CUDA device detected, but this harness has no CUDA launch runner; "
        "the NVVM lowering output is in results/raw/gpu_lowered.mlir. "
        "No GPU timing numbers are reported.\n"
    )
    print(GPU_STATUS.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
