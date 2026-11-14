"""TileFlow IR generation and cost-model schedule selection helpers."""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path

from build_kernel import REPO_ROOT, TILEFLOW_TUNE

SCRATCHPAD_BYTES = 65536
MACHINE = REPO_ROOT / "bench" / "machine.json"


def schedule_attr(tiles, order, promote, unroll=1, target="cpu") -> str:
    tiles_text = ", ".join(str(int(value)) for value in tiles)
    order_text = ", ".join(str(int(value)) for value in order)
    promote_text = ", ".join(f'"{name}"' for name in promote)
    return (
        f"#tileflow.schedule<tiles = [{tiles_text}], order = [{order_text}], "
        f"unroll = {unroll}, promote = [{promote_text}], target = \"{target}\">"
    )


def matmul_source(m, n, k, schedule=None, function="matmul") -> str:
    attribute = f" {{schedule = {schedule}}}" if schedule else ""
    return f"""func.func @{function}(%a: tensor<{m}x{k}xf32>, %b: tensor<{k}x{n}xf32>) -> tensor<{m}x{n}xf32> {{
  %0 = tileflow.matmul %a, %b{attribute} : tensor<{m}x{k}xf32>, tensor<{k}x{n}xf32> -> tensor<{m}x{n}xf32>
  return %0 : tensor<{m}x{n}xf32>
}}
"""


def attention_source(heads, seq, dim, scale, causal, schedule=None,
                     function="attn") -> str:
    attribute = f" {{causal, scale = {scale:.9g}, schedule = {schedule}}} " if schedule else f" {{causal, scale = {scale:.9g}}}"
    if not causal:
        attribute = attribute.replace("causal, ", "")
    return f"""func.func @{function}(%q: tensor<{heads}x{seq}x{dim}xf32>, %k: tensor<{heads}x{seq}x{dim}xf32>, %v: tensor<{heads}x{seq}x{dim}xf32>) -> tensor<{heads}x{seq}x{dim}xf32> {{
  %0 = tileflow.attention %q, %k, %v{attribute}: tensor<{heads}x{seq}x{dim}xf32>, tensor<{heads}x{seq}x{dim}xf32>, tensor<{heads}x{seq}x{dim}xf32> -> tensor<{heads}x{seq}x{dim}xf32>
  return %0 : tensor<{heads}x{seq}x{dim}xf32>
}}
"""


def _schedule_from_winner(winner, promote_names):
    tiles = [int(value) for value in winner["tiles"]]
    order = [int(value) for value in winner["order"]]
    promote = list(promote_names)
    return tiles, order, promote


def _shrink_to_scratchpad(tiles, element_bytes, num_operands, loop_bits):
    """Halve the largest tile until the promoted operands fit the scratchpad."""
    tiles = list(tiles)
    while True:
        footprint = 0
        for bits in loop_bits:
            size = element_bytes
            for index in bits:
                size *= tiles[index]
            footprint += size
        if footprint <= SCRATCHPAD_BYTES:
            return tiles
        largest = max(range(len(tiles)), key=lambda index: tiles[index])
        if tiles[largest] <= 8:
            return tiles
        tiles[largest] //= 2


def tune_schedule(source: str, kind: str, promote: bool, target: str = "cpu"):
    """Run tileflow-tune and return (tiles, order, promote) for the shape."""
    tune_dir = REPO_ROOT / "results" / "raw" / "tune"
    tune_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".mlir", dir=tune_dir, delete=False
    ) as handle:
        handle.write(source)
        source_path = handle.name
    json_path = tune_dir / (Path(source_path).stem + ".json")
    log_path = tune_dir / (Path(source_path).stem + ".log")
    subprocess.run(
        [
            TILEFLOW_TUNE,
            source_path,
            "--machine",
            str(MACHINE),
            "--json",
            str(json_path),
            "--log",
            str(log_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    ranking = json.loads(json_path.read_text())
    winner = ranking["winner"]

    if not promote:
        return _schedule_from_winner(winner, [])

    if kind == "matmul":
        promote_names = ["A", "B"]
        tiles = _shrink_to_scratchpad(winner["tiles"], 4, 2, [[0, 2], [2, 1]])
    elif kind == "attention":
        promote_names = ["K", "V"]
        tiles = _shrink_to_scratchpad(
            winner["tiles"], 4, 2, [[0, 2, 3], [0, 2, 3]]
        )
    else:
        promote_names = []
        tiles = winner["tiles"]
    return tiles, [int(value) for value in winner["order"]], promote_names


def baseline_schedule(extents):
    return schedule_attr(extents, list(range(len(extents))), [])


def matmul_flops(m, n, k):
    return 2.0 * m * n * k


def attention_flops(heads, seq, dim):
    return 4.0 * heads * seq * seq * dim + 5.0 * heads * seq * seq
