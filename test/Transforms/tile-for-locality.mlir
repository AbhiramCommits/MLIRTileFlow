// RUN: tileflow-opt -tileflow-to-linalg -tileflow-tile-for-locality %s | FileCheck %s
//
// Tiling a non-divisible matmul (50x70x30 with 16x16x16 tiles) produces a
// three-deep scf.for nest and affine.min boundary tiles.

// CHECK-LABEL: func.func @non_divisible
// CHECK: scf.for {{.*}} to %c50 step %c16
// CHECK: scf.for {{.*}} to %c30 step %c16
// CHECK: scf.for {{.*}} to %c70 step %c16
// CHECK: affine.min
// CHECK: affine.apply
// CHECK: tensor.extract_slice
// CHECK-SAME: [%{{.*}}, %{{.*}}]
// CHECK-SAME: to tensor<?x?xf32>
// CHECK: linalg.matmul
func.func @non_divisible(%a: tensor<50x70xf32>, %b: tensor<70x30xf32>)
    -> tensor<50x30xf32> {
  %0 = tileflow.matmul %a, %b {
    schedule = #tileflow.schedule<tiles = [16, 16, 16], order = [0, 1, 2],
                                  unroll = 1, promote = [], target = "cpu">
  } : tensor<50x70xf32>, tensor<70x30xf32> -> tensor<50x30xf32>
  return %0 : tensor<50x30xf32>
}
