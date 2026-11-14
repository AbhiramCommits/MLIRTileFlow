#!/usr/bin/env python3
"""Numerical validation of the compiled TileFlow kernels against PyTorch."""

from __future__ import annotations

import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_kernel import REPO_ROOT, build_kernel
from tileflow_ir import (
    attention_source,
    baseline_schedule,
    matmul_source,
    schedule_attr,
    tune_schedule,
)

RESULTS = REPO_ROOT / "results"
NUMERICS_PATH = RESULTS / "numerics.json"

BASELINE_PIPELINE = "-tileflow-to-linalg -tileflow-to-loops"
TILED_PIPELINE = (
    "-tileflow-to-linalg -tileflow-apply-schedule='fuse=true promote=false' "
    "-tileflow-to-loops"
)
PROMOTED_PIPELINE = (
    "-tileflow-to-linalg -tileflow-apply-schedule='fuse=true promote=true' "
    "-tileflow-to-loops='vectorize=true'"
)

VARIANTS = [
    ("untiled", BASELINE_PIPELINE),
    ("tiled_fused", TILED_PIPELINE),
    ("tiled_fused_promoted_vectorized", PROMOTED_PIPELINE),
]


@lru_cache(maxsize=None)
def tuned(kind: str, shape: tuple, promote: bool):
    if kind == "matmul":
        m, n, k = shape
        source = matmul_source(m, n, k)
    else:
        heads, seq, dim = shape
        source = attention_source(heads, seq, dim, 1.0 / math.sqrt(dim), True)
    return tune_schedule(source, kind, promote)


def record(case: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    if NUMERICS_PATH.exists():
        data = json.loads(NUMERICS_PATH.read_text())
    else:
        data = {"tolerance": {"atol": 2e-3, "rtol": 2e-3}, "cases": []}
    data["cases"] = [
        entry
        for entry in data["cases"]
        if (entry["case"], entry["variant"]) != (case["case"], case["variant"])
    ]
    data["cases"].append(case)
    data["cases"].sort(key=lambda entry: (entry["case"], entry["variant"]))
    NUMERICS_PATH.write_text(json.dumps(data, indent=2) + "\n")


def compare(case_name: str, variant: str, actual: np.ndarray,
            expected: np.ndarray, atol: float, rtol: float) -> None:
    difference = np.abs(actual - expected)
    max_abs = float(difference.max()) if difference.size else 0.0
    denominator = np.maximum(np.abs(expected), 1e-6)
    max_rel = float((difference / denominator).max()) if difference.size else 0.0
    passed = bool(np.allclose(actual, expected, atol=atol, rtol=rtol))
    record(
        {
            "case": case_name,
            "variant": variant,
            "max_abs_error": max_abs,
            "max_rel_error": max_rel,
            "atol": atol,
            "rtol": rtol,
            "passed": passed,
        }
    )
    assert passed, (
        f"{case_name}/{variant}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
        f"(atol={atol}, rtol={rtol})"
    )


MATMUL_SHAPES = [
    ("matmul_256", (256, 256, 256)),
    ("matmul_1024", (1024, 1024, 1024)),
    ("matmul_nondivisible", (513, 257, 129)),
]


@pytest.mark.parametrize("name,shape", MATMUL_SHAPES, ids=[s[0] for s in MATMUL_SHAPES])
def test_matmul(name: str, shape: tuple) -> None:
    m, n, k = shape
    rng = np.random.default_rng(0)
    a = rng.standard_normal((m, k), dtype=np.float32) * 0.25
    b = rng.standard_normal((k, n), dtype=np.float32) * 0.25
    expected = torch.matmul(torch.from_numpy(a), torch.from_numpy(b)).numpy()

    tiles, order, _ = tuned("matmul", shape, False)
    promoted_tiles, promoted_order, promote = tuned("matmul", shape, True)

    sources = {
        "untiled": (matmul_source(m, n, k, baseline_schedule([m, n, k])), "matmul"),
        "tiled_fused": (
            matmul_source(
                m, n, k, schedule_attr(tiles, order, []), function="matmul_tiled"
            ),
            "matmul_tiled",
        ),
        "tiled_fused_promoted_vectorized": (
            matmul_source(
                m, n, k, schedule_attr(promoted_tiles, promoted_order, promote),
                function="matmul_promoted",
            ),
            "matmul_promoted",
        ),
    }

    for variant, pipeline in VARIANTS:
        source, function = sources[variant]
        kernel = build_kernel(source, pipeline, function, tag=f"{name}_{variant}")
        actual = kernel(a, b)
        compare(name, variant, actual, expected, atol=2e-3, rtol=2e-3)


ATTENTION_CASES = [
    ("attention_seq128_causal", (1, 128, 64), True),
    ("attention_seq128", (1, 128, 64), False),
    ("attention_seq512_causal", (1, 512, 64), True),
    ("attention_seq512", (1, 512, 64), False),
]


@pytest.mark.parametrize(
    "name,shape,causal", ATTENTION_CASES, ids=[case[0] for case in ATTENTION_CASES]
)
def test_attention(name: str, shape: tuple, causal: bool) -> None:
    heads, seq, dim = shape
    scale = 1.0 / math.sqrt(dim)
    rng = np.random.default_rng(1)
    q = rng.standard_normal((heads, seq, dim), dtype=np.float32) * 0.5
    k = rng.standard_normal((heads, seq, dim), dtype=np.float32) * 0.5
    v = rng.standard_normal((heads, seq, dim), dtype=np.float32) * 0.5

    reference = torch.nn.functional.scaled_dot_product_attention(
        torch.from_numpy(q).unsqueeze(1),
        torch.from_numpy(k).unsqueeze(1),
        torch.from_numpy(v).unsqueeze(1),
        is_causal=causal,
        scale=scale,
    ).squeeze(1).numpy()

    tiles, order, _ = tuned("attention", shape, False)
    promoted_tiles, promoted_order, promote = tuned("attention", shape, True)

    sources = {
        "untiled": (
            attention_source(
                heads, seq, dim, scale, causal,
                baseline_schedule([heads, seq, seq, dim]),
            ),
            "attn",
        ),
        "tiled_fused": (
            attention_source(
                heads, seq, dim, scale, causal, schedule_attr(tiles, order, []),
                function="attn_tiled",
            ),
            "attn_tiled",
        ),
        "tiled_fused_promoted_vectorized": (
            attention_source(
                heads, seq, dim, scale, causal,
                schedule_attr(promoted_tiles, promoted_order, promote),
                function="attn_promoted",
            ),
            "attn_promoted",
        ),
    }

    for variant, pipeline in VARIANTS:
        source, function = sources[variant]
        kernel = build_kernel(source, pipeline, function, tag=f"{name}_{variant}")
        actual = kernel(q, k, v)
        compare(name, variant, actual, reference, atol=2e-3, rtol=2e-3)
