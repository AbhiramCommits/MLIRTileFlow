// RUN: tileflow-opt %s | tileflow-opt | FileCheck %s
//
// Every operation and the schedule attribute must round-trip through the
// parser and printer twice.

// CHECK-LABEL: func.func @matmul
// CHECK: tileflow.matmul
// CHECK-SAME: schedule = #tileflow.schedule<tiles = [32, 32, 64], order = [0, 1, 2], unroll = 2, promote = ["A", "B"], target = "cpu">
// CHECK-SAME: tensor<128x256xf32>, tensor<256x512xf32> -> tensor<128x512xf32>
func.func @matmul(%a: tensor<128x256xf32>, %b: tensor<256x512xf32>)
    -> tensor<128x512xf32> {
  %0 = tileflow.matmul %a, %b {
    schedule = #tileflow.schedule<tiles = [32, 32, 64], order = [0, 1, 2],
                                  unroll = 2, promote = ["A", "B"],
                                  target = "cpu">
  } : tensor<128x256xf32>, tensor<256x512xf32> -> tensor<128x512xf32>
  return %0 : tensor<128x512xf32>
}

// CHECK-LABEL: func.func @matmul_transposed
// CHECK: tileflow.matmul
// CHECK-SAME: {transpose_a, transpose_b}
func.func @matmul_transposed(%a: tensor<256x128xf32>, %b: tensor<512x256xf32>)
    -> tensor<128x512xf32> {
  %0 = tileflow.matmul %a, %b {transpose_a, transpose_b}
      : tensor<256x128xf32>, tensor<512x256xf32> -> tensor<128x512xf32>
  return %0 : tensor<128x512xf32>
}

// CHECK-LABEL: func.func @softmax
// CHECK: tileflow.softmax_row
// CHECK-SAME: {axis = 0 : i64}
// CHECK-SAME: : tensor<64x128xf32>
func.func @softmax(%x: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = tileflow.softmax_row %x {axis = 0} : tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @attention
// CHECK: tileflow.attention
// CHECK-SAME: {causal, scale = 1.250000e-01 : f64, schedule = #tileflow.schedule<tiles = [1, 32, 32, 64], order = [0, 1, 2, 3], unroll = 1, promote = ["K", "V"], target = "gpu">}
func.func @attention(%q: tensor<2x128x64xf32>, %k: tensor<2x128x64xf32>,
                     %v: tensor<2x128x64xf32>) -> tensor<2x128x64xf32> {
  %0 = tileflow.attention %q, %k, %v {
    causal, scale = 0.125,
    schedule = #tileflow.schedule<tiles = [1, 32, 32, 64], order = [0, 1, 2, 3],
                                  unroll = 1, promote = ["K", "V"],
                                  target = "gpu">
  } : tensor<2x128x64xf32>, tensor<2x128x64xf32>, tensor<2x128x64xf32>
      -> tensor<2x128x64xf32>
  return %0 : tensor<2x128x64xf32>
}

// CHECK-LABEL: func.func @region
// CHECK: tileflow.schedule_region #tileflow.schedule<tiles = [4, 4], order = [0, 1], unroll = 1, promote = [], target = "cpu">
// CHECK:   tileflow.softmax_row
// CHECK:   tileflow.yield
func.func @region(%x: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = tileflow.schedule_region #tileflow.schedule<tiles = [4, 4],
      order = [0, 1], unroll = 1, promote = [], target = "cpu"> {
    %1 = tileflow.softmax_row %x : tensor<8x8xf32>
    tileflow.yield %1 : tensor<8x8xf32>
  } : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
