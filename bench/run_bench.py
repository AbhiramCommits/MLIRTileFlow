#!/usr/bin/env python3
"""CPU benchmarks: tiled, fused, promoted and vectorized variants vs untiled."""

from __future__ import annotations

import csv
import json
import math
import platform
import shutil
import statistics
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from build_kernel import REPO_ROOT, TILEFLOW_OPT, build_kernel
from tileflow_ir import (
    attention_flops,
    attention_source,
    baseline_schedule,
    matmul_flops,
    matmul_source,
    schedule_attr,
    tune_schedule,
)

RESULTS = REPO_ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

BASELINE_PIPELINE = "-tileflow-to-linalg -tileflow-to-loops"
TILED_PIPELINE = (
    "-tileflow-to-linalg -tileflow-apply-schedule='fuse=true promote=false' "
    "-tileflow-to-loops"
)
PROMOTED_PIPELINE = (
    "-tileflow-to-linalg -tileflow-apply-schedule='fuse=true promote=true' "
    "-tileflow-to-loops='vectorize=true'"
)

REPS = 20
WARMUP = 3


@lru_cache(maxsize=None)
def tuned(kind: str, shape: tuple, promote: bool):
    if kind == "matmul":
        m, n, k = shape
        source = matmul_source(m, n, k)
    else:
        heads, seq, dim = shape
        source = attention_source(heads, seq, dim, 1.0 / math.sqrt(dim), True)
    return tune_schedule(source, kind, promote)


def predicted_dram(source: str) -> int | None:
    completed = subprocess.run(
        [TILEFLOW_OPT, "-tileflow-locality-report", "-o", "/dev/null"],
        input=source,
        text=True,
        capture_output=True,
        check=True,
    )
    line = completed.stdout.splitlines()[0]
    report = json.loads(line)
    if not report["ops"]:
        return None
    return int(report["ops"][0]["dram_bytes"])


def benchmark(kernel, inputs, reps: int = REPS) -> tuple[float, float, float]:
    for _ in range(WARMUP):
        kernel(*inputs)
    timings = []
    for _ in range(reps):
        start = time.perf_counter()
        kernel(*inputs)
        timings.append((time.perf_counter() - start) * 1e3)
    return (
        float(np.median(timings)),
        float(np.percentile(timings, 10)),
        float(np.percentile(timings, 90)),
    )


def build_variant(name: str, kind: str, shape: tuple, variant: str):
    if kind == "matmul":
        m, n, k = shape
        tiles, order, _ = tuned(kind, shape, False)
        promoted_tiles, promoted_order, promote = tuned(kind, shape, True)
        if variant == "untiled":
            source = matmul_source(
                m, n, k, baseline_schedule([m, n, k]), function=f"{name}_untiled"
            )
            pipeline = BASELINE_PIPELINE
            function = f"{name}_untiled"
            flops = matmul_flops(m, n, k)
        elif variant == "tiled_fused":
            source = matmul_source(
                m, n, k, schedule_attr(tiles, order, []), function=f"{name}_tiled"
            )
            pipeline = TILED_PIPELINE
            function = f"{name}_tiled"
            flops = matmul_flops(m, n, k)
        else:
            source = matmul_source(
                m, n, k, schedule_attr(promoted_tiles, promoted_order, promote),
                function=f"{name}_promoted",
            )
            pipeline = PROMOTED_PIPELINE
            function = f"{name}_promoted"
            flops = matmul_flops(m, n, k)
        inputs = (
            np.random.default_rng(0).standard_normal((m, k), dtype=np.float32) * 0.25,
            np.random.default_rng(1).standard_normal((k, n), dtype=np.float32) * 0.25,
        )
    else:
        heads, seq, dim = shape
        scale = 1.0 / math.sqrt(dim)
        tiles, order, _ = tuned(kind, shape, False)
        promoted_tiles, promoted_order, promote = tuned(kind, shape, True)
        if variant == "untiled":
            source = attention_source(
                heads, seq, dim, scale, True,
                baseline_schedule([heads, seq, seq, dim]),
                function=f"{name}_untiled",
            )
            pipeline = BASELINE_PIPELINE
            function = f"{name}_untiled"
        elif variant == "tiled_fused":
            source = attention_source(
                heads, seq, dim, scale, True, schedule_attr(tiles, order, []),
                function=f"{name}_tiled",
            )
            pipeline = TILED_PIPELINE
            function = f"{name}_tiled"
        else:
            source = attention_source(
                heads, seq, dim, scale, True,
                schedule_attr(promoted_tiles, promoted_order, promote),
                function=f"{name}_promoted",
            )
            pipeline = PROMOTED_PIPELINE
            function = f"{name}_promoted"
        flops = attention_flops(heads, seq, dim)
        rng = np.random.default_rng(2)
        inputs = tuple(
            rng.standard_normal((heads, seq, dim), dtype=np.float32) * 0.5
            for _ in range(3)
        )

    kernel = build_kernel(source, pipeline, function, tag=f"{name}_{variant}")
    return kernel, inputs, flops, predicted_dram(source)


def spearman(xs, ys) -> float:
    def ranks(values):
        order = sorted(range(len(values)), key=lambda index: values[index])
        result = [0.0] * len(values)
        position = 0
        while position < len(order):
            end = position
            while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
                end += 1
            average = (position + end) / 2.0 + 1.0
            for index in range(position, end + 1):
                result[order[index]] = average
            position = end + 1
        return result

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mean_x, mean_y = statistics.mean(rx), statistics.mean(ry)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    denominator = math.sqrt(
        sum((a - mean_x) ** 2 for a in rx) * sum((b - mean_y) ** 2 for b in ry)
    )
    return numerator / denominator if denominator else 0.0


def model_vs_measured() -> dict:
    kind = "matmul"
    shape = (512, 512, 512)
    m, n, k = shape
    source = matmul_source(m, n, k)
    tune_dir = RESULTS / "raw" / "tune"
    tune_dir.mkdir(parents=True, exist_ok=True)
    tune_source = tune_dir / "bench_candidates.mlir"
    tune_source.write_text(source)
    json_path = tune_dir / "bench_candidates.json"
    log_path = tune_dir / "bench_candidates.log"
    subprocess.run(
        [
            str(REPO_ROOT / "build" / "bin" / "tileflow-tune"),
            str(tune_source),
            "--machine",
            str(REPO_ROOT / "bench" / "machine.json"),
            "--json",
            str(json_path),
            "--log",
            str(log_path),
            "--top",
            "200",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    ranking = json.loads(json_path.read_text())

    # A curated candidate set spanning no tiling through small tiles, so the
    # comparison covers a real traffic range instead of permutations that all
    # predict the same number.
    curated = [
        [512, 512, 512],
        [256, 256, 256],
        [128, 128, 128],
        [64, 64, 64],
        [32, 32, 32],
        [64, 64, 256],
        [64, 256, 64],
        [256, 64, 64],
    ]
    candidates = [
        {"tiles": tiles, "order": [0, 1, 2], "rank": index + 1}
        for index, tiles in enumerate(curated)
    ]

    a = np.random.default_rng(10).standard_normal((m, k), dtype=np.float32) * 0.25
    b = np.random.default_rng(11).standard_normal((k, n), dtype=np.float32) * 0.25
    records = []
    for index, entry in enumerate(candidates):
        candidate_source = matmul_source(
            m, n, k, schedule_attr(entry["tiles"], entry["order"], []),
            function=f"cand_{index}",
        )
        kernel = build_kernel(
            candidate_source, TILED_PIPELINE, f"cand_{index}", tag=f"cand_{index}"
        )
        median, _, _ = benchmark(kernel, (a, b), reps=7)
        records.append(
            {
                "tiles": entry["tiles"],
                "order": entry["order"],
                "predicted_dram_bytes": predicted_dram(candidate_source),
                "median_ms": median,
            }
        )

    predicted = [record["predicted_dram_bytes"] for record in records]
    measured = [record["median_ms"] for record in records]
    rho = spearman(predicted, measured)
    model_top = min(records, key=lambda record: record["predicted_dram_bytes"])
    best_measured = min(records, key=lambda record: record["median_ms"])
    regret = model_top["median_ms"] / best_measured["median_ms"] - 1.0

    perf_available = shutil.which("perf") is not None
    output = {
        "shape": list(shape),
        "candidate_count": len(records),
        "spearman_predicted_traffic_vs_runtime": rho,
        "model_top1": model_top,
        "best_measured": best_measured,
        "top1_regret": regret,
        "traffic_source": "model-estimated",
        "runtime_source": "measured (perf_counter, median of 7)",
        "perf_stat_available": perf_available,
        "llc_misses_measured": False,
        "note": (
            "perf stat with cache/DRAM counters is not available on this machine; "
            "traffic figures are model estimates and only runtime is measured."
            if not perf_available
            else "perf available; cache counter collection was not required for this run."
        ),
        "candidates": records,
    }
    (RESULTS / "model_vs_measured.json").write_text(json.dumps(output, indent=2) + "\n")
    return output


def main() -> int:
    import os

    quick = os.environ.get("TILEFLOW_BENCH_QUICK") == "1"
    if quick:
        cases = [
            ("matmul_256", "matmul", (256, 256, 256)),
            ("attention_seq128", "attention", (1, 128, 64)),
        ]
    else:
        cases = [
            ("matmul_256", "matmul", (256, 256, 256)),
            ("matmul_1024", "matmul", (1024, 1024, 1024)),
            ("matmul_nondivisible", "matmul", (513, 257, 129)),
            ("attention_seq128", "attention", (1, 128, 64)),
            ("attention_seq512", "attention", (1, 512, 64)),
        ]
    variants = ["untiled", "tiled_fused", "tiled_fused_promoted_vectorized"]

    rows = []
    for name, kind, shape in cases:
        baseline_ms = None
        for variant in variants:
            kernel, inputs, flops, dram = build_variant(name, kind, shape, variant)
            median, p10, p90 = benchmark(kernel, inputs)
            if variant == "untiled":
                baseline_ms = median
            gflops = flops / (median * 1e-3) / 1e9
            speedup = (baseline_ms / median) if baseline_ms else 1.0
            rows.append(
                {
                    "shape": name,
                    "variant": variant,
                    "median_ms": median,
                    "p10_ms": p10,
                    "p90_ms": p90,
                    "gflops": gflops,
                    "speedup_vs_untiled": speedup,
                    "predicted_dram_bytes": dram,
                    "traffic_source": "model-estimated",
                }
            )
            print(
                f"{name:22s} {variant:38s} {median:9.3f} ms "
                f"{gflops:7.2f} GFLOP/s {speedup:5.2f}x dram={dram}"
            )

    with (RESULTS / "bench_cpu.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# CPU benchmarks",
        "",
        f"Machine: {platform.platform()} / {platform.processor()}",
        "",
        "| shape | variant | median ms | p10 ms | p90 ms | GFLOP/s | speedup | predicted DRAM bytes (model) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {shape} | {variant} | {median_ms:.3f} | {p10_ms:.3f} | {p90_ms:.3f} | "
            "{gflops:.2f} | {speedup_vs_untiled:.2f}x | {predicted_dram_bytes} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "`predicted DRAM bytes` is produced by the analytical cost model;",
            "runtime columns are measured with `time.perf_counter` (median of 20).",
            "Vectorization applies to elementwise linalg generics only; matmul",
            "vectorization was intentionally skipped for these tile sizes.",
            "",
        ]
    )
    (RESULTS / "bench_cpu.md").write_text("\n".join(lines))

    model = model_vs_measured()
    print(
        "model vs measured: spearman={:.3f} regret={:.3f}".format(
            model["spearman_predicted_traffic_vs_runtime"], model["top1_regret"]
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
