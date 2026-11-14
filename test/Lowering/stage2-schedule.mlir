// RUN: tileflow-opt -tileflow-to-linalg -tileflow-apply-schedule='tile-sizes=16,16,16 order=0,1,2 fuse=true promote=true' %s | FileCheck %s
//
// Stage 2: the schedule is applied on tensors, then the module is bufferized
// and the A/B operand tiles are copied into private scratchpads.

// CHECK-LABEL: func.func @matmul
// CHECK-DAG: memref<64x64xf32>
// CHECK: memref.alloca() : memref<16x16xf32>
// CHECK: memref.alloca() : memref<16x16xf32>
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: memref.subview
// CHECK: linalg.copy
// CHECK: linalg.copy
// CHECK: linalg.matmul
// CHECK-SAME: memref<16x16xf32>, memref<16x16xf32>
// CHECK-NOT: tensor<
func.func @matmul(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>)
    -> tensor<64x64xf32> {
  %0 = tileflow.matmul %a, %b {
    schedule = #tileflow.schedule<tiles = [16, 16, 16], order = [0, 1, 2],
                                  unroll = 1, promote = ["A", "B"],
                                  target = "cpu">
  } : tensor<64x64xf32>, tensor<64x64xf32> -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}
