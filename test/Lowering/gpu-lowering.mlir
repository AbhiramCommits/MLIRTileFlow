// RUN: tileflow-opt -tileflow-lower-to-gpu='stop-after=forall' %s | FileCheck %s --check-prefix=FORALL
// RUN: tileflow-opt -tileflow-lower-to-gpu='stop-after=gpu' %s | FileCheck %s --check-prefix=GPU
// RUN: tileflow-opt -tileflow-lower-to-gpu='stop-after=nvvm' %s | FileCheck %s --check-prefix=NVVM
//
// The GPU path maps the parallel tile loops of the schedule to scf.forall with
// GPU mappings, then to gpu.launch, outlines a kernel and converts the kernel
// body to NVVM. It is compile/FileCheck-tested; execution requires a CUDA GPU.

// FORALL-LABEL: func.func @matmul
// FORALL: scf.forall
// FORALL: mapping = [#gpu.block<x>, #gpu.thread<x>]

// GPU-LABEL: func.func @matmul
// GPU: gpu.launch
// GPU: gpu.terminator
// GPU-NOT: scf.parallel

// NVVM: gpu.launch_func
// NVVM-LABEL: gpu.module @matmul_kernel
// NVVM: nvvm.read.ptx.sreg.ctaid.x
func.func @matmul(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>)
    -> tensor<64x64xf32> {
  %0 = tileflow.matmul %a, %b {
    schedule = #tileflow.schedule<tiles = [16, 16, 16], order = [0, 1, 2],
                                  unroll = 1, promote = ["A", "B"],
                                  target = "gpu">
  } : tensor<64x64xf32>, tensor<64x64xf32> -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}
