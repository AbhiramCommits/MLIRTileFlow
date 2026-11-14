// RUN: tileflow-opt -tileflow-to-linalg -tileflow-tile-for-locality -tileflow-promote-to-scratchpad='operands=A,B' %s | FileCheck %s --check-prefix=CPU
// RUN: tileflow-opt -tileflow-to-linalg -tileflow-tile-for-locality -tileflow-promote-to-scratchpad='space=workgroup operands=A,B' %s | FileCheck %s --check-prefix=GPU
//
// CPU promotion allocates private stack scratchpads with copy-in/copy-out;
// the GPU space allocates workgroup memory (memref memory space 3).
//
// The schedule only promotes A and B, so exactly two scratchpads are created.

// CPU-LABEL: func.func @matmul
// CPU: memref.alloca() : memref<16x16xf32>
// CPU: memref.alloca() : memref<16x16xf32>
// CPU: scf.for
// CPU: linalg.copy
// CPU: linalg.copy
// CPU: linalg.matmul
// CPU-SAME: memref<16x16xf32>, memref<16x16xf32>
// CPU-NOT: memref<16x16xf32, 3>

// GPU-LABEL: func.func @matmul
// GPU: gpu.alloc  () : memref<16x16xf32, 3>
// GPU: gpu.alloc  () : memref<16x16xf32, 3>
// GPU: scf.for
// GPU: linalg.copy
// GPU: linalg.copy
// GPU: linalg.matmul
func.func @matmul(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>)
    -> tensor<64x64xf32> {
  %0 = tileflow.matmul %a, %b {
    schedule = #tileflow.schedule<tiles = [16, 16, 16], order = [0, 1, 2],
                                  unroll = 1, promote = ["A", "B"],
                                  target = "cpu">
  } : tensor<64x64xf32>, tensor<64x64xf32> -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}
