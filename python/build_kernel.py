#!/usr/bin/env python3
"""Compile TileFlow MLIR to a shared library and call it through ctypes.

The pipeline always ends with `-llvm-request-c-wrappers -tileflow-to-llvm`,
so the exported entry point is the MLIR C interface wrapper
`_mlir_ciface_<function>`. Memrefs cross the boundary as strided memref
descriptors (allocated, aligned, offset, sizes..., strides...); results come
first in the wrapper signature, followed by the input memrefs.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_RAW = REPO_ROOT / "results" / "raw" / "kernels"


def _tool(env_name: str, default: str) -> str:
    return os.environ.get(env_name, str(REPO_ROOT / "build" / "bin" / default))


TILEFLOW_OPT = _tool("TILEFLOW_OPT", "tileflow-opt")
TILEFLOW_TUNE = _tool("TILEFLOW_TUNE", "tileflow-tune")
def _default_mlir_translate() -> str:
    candidates = [
        shutil.which("mlir-translate"),
        "/opt/homebrew/opt/llvm/bin/mlir-translate",
        "/usr/lib/llvm-23/bin/mlir-translate",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return "mlir-translate"


MLIR_TRANSLATE = os.environ.get("MLIR_TRANSLATE", _default_mlir_translate())
def _default_clang() -> str:
    # The .ll files come from MLIR_TRANSLATE, so compile them with the clang from
    # the same LLVM install. A system clang from an older LLVM (e.g. Ubuntu's
    # /usr/bin/clang 18 next to llvm-23's mlir-translate) cannot parse newer IR.
    sibling = Path(MLIR_TRANSLATE).parent / "clang"
    candidates = [
        str(sibling) if Path(MLIR_TRANSLATE).is_absolute() else None,
        "/opt/homebrew/opt/llvm/bin/clang",
        shutil.which("clang"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return "clang"


CLANG = os.environ.get("CLANG", _default_clang())

_DESCRIPTOR_CACHE: dict[int, type[ctypes.Structure]] = {}
_LIBRARY_CACHE: dict[str, ctypes.CDLL] = {}
_LIBC = ctypes.CDLL(None)
_LIBC.free.argtypes = [ctypes.c_void_p]


def _run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, capture_output=True, text=True, **kwargs)


def descriptor_type(rank: int) -> type[ctypes.Structure]:
    if rank in _DESCRIPTOR_CACHE:
        return _DESCRIPTOR_CACHE[rank]
    fields = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("sizes", ctypes.c_int64 * max(rank, 1)),
        ("strides", ctypes.c_int64 * max(rank, 1)),
    ]
    descriptor = type(f"MemRefDescriptor{rank}", (ctypes.Structure,), {"_fields_": fields})
    _DESCRIPTOR_CACHE[rank] = descriptor
    return descriptor


def make_descriptor(array: np.ndarray):
    descriptor = descriptor_type(array.ndim)()
    descriptor.allocated = array.ctypes.data
    descriptor.aligned = array.ctypes.data
    descriptor.offset = 0
    for index in range(array.ndim):
        descriptor.sizes[index] = array.shape[index]
        descriptor.strides[index] = array.strides[index] // array.itemsize
    return descriptor


class CompiledKernel:
    def __init__(self, library: ctypes.CDLL, function_name: str, source_path: Path):
        self.library = library
        self.function_name = function_name
        self.source_path = source_path
        symbol = f"_mlir_ciface_{function_name}"
        if not hasattr(library, symbol):
            symbol = function_name
        self.symbol_name = symbol
        self.function = getattr(library, symbol)
        self.function.restype = None

    def __call__(self, *inputs: np.ndarray) -> np.ndarray:
        arrays = []
        descriptors = []
        for value in inputs:
            array = np.ascontiguousarray(value, dtype=np.float32)
            arrays.append(array)
            descriptors.append(make_descriptor(array))

        result_descriptor = descriptor_type(arrays[0].ndim)()
        argument_types = [ctypes.POINTER(type(result_descriptor))] * (
            len(descriptors) + 1
        )
        self.function.argtypes = argument_types
        self.function(
            ctypes.byref(result_descriptor),
            *[ctypes.byref(descriptor) for descriptor in descriptors],
        )

        rank = arrays[0].ndim
        sizes = tuple(int(result_descriptor.sizes[i]) for i in range(rank))
        strides = tuple(int(result_descriptor.strides[i]) * 4 for i in range(rank))
        total = int(np.prod(sizes))
        base = result_descriptor.aligned + result_descriptor.offset * 4
        if total:
            buffer = (ctypes.c_float * total).from_address(base)
            result = np.ndarray(sizes, dtype=np.float32, buffer=buffer, strides=strides).copy()
        else:
            result = np.empty(sizes, dtype=np.float32)

        if result_descriptor.allocated:
            _LIBC.free(ctypes.c_void_p(result_descriptor.allocated))
        return result


def _pipeline_arguments(pipeline: str | list[str]) -> list[str]:
    arguments = shlex.split(pipeline) if isinstance(pipeline, str) else list(pipeline)
    # Insert the C wrapper request immediately before the LLVM stage.
    if "-tileflow-to-llvm" in arguments:
        index = arguments.index("-tileflow-to-llvm")
        arguments.insert(index, "-llvm-request-c-wrappers")
    else:
        arguments.extend(["-llvm-request-c-wrappers", "-tileflow-to-llvm"])
    return arguments


def build_kernel(
    mlir_source: str,
    pipeline: str | list[str],
    function_name: str,
    tag: str = "",
) -> CompiledKernel:
    """Compile one MLIR function and return a callable kernel."""
    RESULTS_RAW.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(
        (mlir_source + "|" + str(pipeline) + "|" + function_name).encode()
    ).hexdigest()[:20]
    base = RESULTS_RAW / f"{tag or function_name}_{key}"
    mlir_path = base.with_suffix(".mlir")
    ll_path = base.with_suffix(".ll")
    so_path = base.with_suffix(".so")

    mlir_path.write_text(mlir_source)
    lowered_path = base.with_suffix(".lowered.mlir")
    if not so_path.exists() or os.environ.get("TILEFLOW_REBUILD") == "1":
        arguments = _pipeline_arguments(pipeline)
        result = _run([TILEFLOW_OPT, str(mlir_path), *arguments])
        lowered_path.write_text(result.stdout)
        _run(
            [MLIR_TRANSLATE, "--mlir-to-llvmir", str(lowered_path), "-o", str(ll_path)]
        )
        _run([CLANG, "-O3", "-shared", "-fPIC", str(ll_path), "-o", str(so_path)])

    key_string = str(so_path)
    if key_string not in _LIBRARY_CACHE:
        _LIBRARY_CACHE[key_string] = ctypes.CDLL(key_string)
    return CompiledKernel(_LIBRARY_CACHE[key_string], function_name, mlir_path)
