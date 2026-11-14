// RUN: tileflow-opt -tileflow-to-linalg %s | FileCheck %s
//
// Stage 1: every tileflow op becomes linalg, and every produced linalg op
// carries the source schedule as a discardable `tileflow.schedule` attribute.

// CHECK-LABEL: func.func @matmul
// CHECK: linalg.matmul
// CHECK-SAME: tileflow.schedule = #tileflow.schedule<tiles = [16, 16, 8], order = [0, 1, 2], unroll = 1, promote = ["A", "B"], target = "cpu">
// CHECK-SAME: ins(%{{.*}}, %{{.*}} : tensor<64x64xf32>, tensor<64x8xf32>)
// CHECK-SAME: outs(%{{.*}} : tensor<64x8xf32>)
func.func @matmul(%a: tensor<64x64xf32>, %b: tensor<64x8xf32>)
    -> tensor<64x8xf32> {
  %0 = tileflow.matmul %a, %b {
    schedule = #tileflow.schedule<tiles = [16, 16, 8], order = [0, 1, 2],
                                  unroll = 1, promote = ["A", "B"],
                                  target = "cpu">
  } : tensor<64x64xf32>, tensor<64x8xf32> -> tensor<64x8xf32>
  return %0 : tensor<64x8xf32>
}

// CHECK-LABEL: func.func @softmax
// CHECK: linalg.fill
// CHECK: linalg.generic
// CHECK: arith.maximumf
// CHECK: linalg.generic
// CHECK: math.exp
// CHECK: linalg.generic
// CHECK: arith.addf
// CHECK: linalg.generic
// CHECK: arith.divf
func.func @softmax(%x: tensor<32x128xf32>) -> tensor<32x128xf32> {
  %0 = tileflow.softmax_row %x
      : tensor<32x128xf32>
  return %0 : tensor<32x128xf32>
}

// CHECK-LABEL: func.func @attention
// CHECK: linalg.transpose
// CHECK: linalg.batch_matmul
// CHECK: linalg.generic
// CHECK: linalg.index 1
// CHECK: linalg.index 2
// CHECK: math.exp
// CHECK: linalg.batch_matmul
// CHECK-SAME: tensor<4x64x64xf32>
func.func @attention(%q: tensor<4x64x64xf32>, %k: tensor<4x64x64xf32>,
                     %v: tensor<4x64x64xf32>) -> tensor<4x64x64xf32> {
  %0 = tileflow.attention %q, %k, %v {causal, scale = 0.125}
      : tensor<4x64x64xf32>, tensor<4x64x64xf32>, tensor<4x64x64xf32>
      -> tensor<4x64x64xf32>
  return %0 : tensor<4x64x64xf32>
}
