// RUN: tileflow-opt -tileflow-to-linalg -tileflow-apply-schedule='tile-sizes=16,16,16 order=0,1,2 fuse=true promote=true' -tileflow-to-loops %s | FileCheck %s
//
// Stage 3: linalg ops are gone, replaced by scf.for loops with scalar
// memref.load/store traffic.

// CHECK-LABEL: func.func @matmul
// CHECK: memref.alloc
// CHECK: scf.for
// CHECK: memref.subview
// CHECK: memref.load
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: memref.store
// CHECK-NOT: linalg.
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
