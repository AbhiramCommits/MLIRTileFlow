#!/usr/bin/env bash
# Resolve a usable MLIR development installation and report the toolchain.
#
# Resolution order for MLIR_DIR:
#   1. $MLIR_DIR
#   2. $(brew --prefix llvm)/lib/cmake/mlir
#   3. /usr/lib/llvm-*/lib/cmake/mlir
#   4. $(llvm-config --cmakedir)
#
# Prints MLIR_DIR, the LLVM version and the paths to mlir-tblgen, mlir-opt,
# mlir-translate and lit. Exits non-zero with install instructions when no
# MLIR installation can be found.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

info() { printf '%s\n' "$*"; }
die() {
  printf 'ERROR: %s\n\n' "$*" >&2
  cat >&2 <<'INSTALL'
Install MLIR/LLVM and re-run scripts/bootstrap.sh:

  macOS (Homebrew):
    brew install llvm cmake ninja
    export MLIR_DIR="$(brew --prefix llvm)/lib/cmake/mlir"
    export PATH="$(brew --prefix llvm)/bin:$PATH"

  Ubuntu 22.04/24.04:
    wget https://apt.llvm.org/llvm.sh
    chmod +x llvm.sh
    sudo ./llvm.sh 20 all            # use the major version you want
    sudo apt-get install -y libmlir-20-dev mlir-20-tools llvm-20-dev
    export MLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir
    export PATH=/usr/lib/llvm-20/bin:$PATH

  Fallback: point MLIR_DIR at a build/install tree containing MLIRConfig.cmake.
INSTALL
  exit 1
}

find_mlir_dir() {
  if [[ -n "${MLIR_DIR:-}" ]]; then
    if [[ -f "${MLIR_DIR}/MLIRConfig.cmake" ]]; then
      printf '%s\n' "${MLIR_DIR}"
      return
    fi
    printf 'WARNING: MLIR_DIR=%s does not contain MLIRConfig.cmake\n' "$MLIR_DIR" >&2
  fi

  if command -v brew >/dev/null 2>&1; then
    local brew_prefix
    brew_prefix="$(brew --prefix llvm 2>/dev/null || true)"
    if [[ -n "$brew_prefix" && -f "$brew_prefix/lib/cmake/mlir/MLIRConfig.cmake" ]]; then
      printf '%s\n' "$brew_prefix/lib/cmake/mlir"
      return
    fi
  fi

  local d
  for d in /usr/lib/llvm-*/lib/cmake/mlir; do
    if [[ -f "$d/MLIRConfig.cmake" ]]; then
      printf '%s\n' "$d"
      return
    fi
  done

  if command -v llvm-config >/dev/null 2>&1; then
    local llvm_cmake
    llvm_cmake="$(llvm-config --cmakedir 2>/dev/null || true)"
    if [[ -n "$llvm_cmake" && -f "$llvm_cmake/MLIRConfig.cmake" ]]; then
      printf '%s\n' "$llvm_cmake"
      return
    fi
    # llvm-config --cmakedir may point at .../lib/cmake/llvm; try the sibling.
    if [[ -n "$llvm_cmake" && -f "${llvm_cmake%/llvm}/mlir/MLIRConfig.cmake" ]]; then
      printf '%s\n' "${llvm_cmake%/llvm}/mlir"
      return
    fi
  fi

  return 1
}

RESOLVED_MLIR_DIR="$(find_mlir_dir || true)"
[[ -n "$RESOLVED_MLIR_DIR" ]] || die "no MLIR development installation found"

# Derive the LLVM install prefix from the MLIR cmake directory.
MLIR_DIR_REAL="$(cd "$RESOLVED_MLIR_DIR" && pwd)"
LLVM_PREFIX="$(cd "$MLIR_DIR_REAL/../../.." && pwd)"

find_tool() {
  local name="$1"
  if [[ -x "$LLVM_PREFIX/bin/$name" ]]; then
    printf '%s\n' "$LLVM_PREFIX/bin/$name"
    return 0
  fi
  if command -v "$name" >/dev/null 2>&1; then
    command -v "$name"
    return 0
  fi
  return 1
}

MLIR_TBLGEN="$(find_tool mlir-tblgen || true)"
MLIR_OPT="$(find_tool mlir-opt || true)"
MLIR_TRANSLATE="$(find_tool mlir-translate || true)"
LLVM_CONFIG="$(find_tool llvm-config || true)"

[[ -n "$MLIR_TBLGEN" ]] || die "mlir-tblgen not found next to $MLIR_DIR_REAL"
[[ -n "$MLIR_OPT" ]] || die "mlir-opt not found next to $MLIR_DIR_REAL"
[[ -n "$MLIR_TRANSLATE" ]] || die "mlir-translate not found next to $MLIR_DIR_REAL"

# lit is usually a Python package, not part of the LLVM binary distribution.
LIT=""
if [[ -x "$ROOT_DIR/.venv/bin/lit" ]]; then
  LIT="$ROOT_DIR/.venv/bin/lit"
elif command -v lit >/dev/null 2>&1; then
  LIT="$(command -v lit)"
elif [[ -x "$LLVM_PREFIX/bin/lit" ]]; then
  LIT="$LLVM_PREFIX/bin/lit"
elif python3 -m lit --version >/dev/null 2>&1; then
  LIT="$(command -v python3) -m lit"
fi
[[ -n "$LIT" ]] || die "lit not found; install it with: python3 -m pip install lit"

LLVM_VERSION="$("$LLVM_CONFIG" --version 2>/dev/null || true)"
if [[ -z "$LLVM_VERSION" ]]; then
  LLVM_VERSION="unknown"
fi

info "MLIR_DIR=$MLIR_DIR_REAL"
info "LLVM_PREFIX=$LLVM_PREFIX"
info "LLVM_VERSION=$LLVM_VERSION"
info "MLIR_TABLEGEN=$MLIR_TBLGEN"
info "MLIR_OPT=$MLIR_OPT"
info "MLIR_TRANSLATE=$MLIR_TRANSLATE"
info "LLVM_CONFIG=$LLVM_CONFIG"
info "LIT=$LIT"
