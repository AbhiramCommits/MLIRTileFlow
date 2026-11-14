// RUN: tileflow-opt -canonicalize %s | FileCheck %s
//
// One CHECK group per canonicalization pattern.

//===----------------------------------------------------------------------===//
// 1. linalg.transpose feeding a matmul is folded into transpose_a.
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func.func @fold_transpose_a
// CHECK: tileflow.matmul %{{.*}}, %{{.*}} {transpose_a}
// CHECK-SAME: : tensor<4x8xf32>, tensor<4x16xf32> -> tensor<8x16xf32>
func.func @fold_transpose_a(%a: tensor<4x8xf32>, %b: tensor<4x16xf32>)
    -> tensor<8x16xf32> {
  %empty = tensor.empty() : tensor<8x4xf32>
  %t = linalg.transpose ins(%a : tensor<4x8xf32>) outs(%empty : tensor<8x4xf32>)
      permutation = [1, 0]
  %0 = tileflow.matmul %t, %b : tensor<8x4xf32>, tensor<4x16xf32>
      -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

//===----------------------------------------------------------------------===//
// 2. An identity schedule_region is erased and its body inlined.
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func.func @erase_identity_region
// CHECK-NOT: tileflow.schedule_region
// CHECK: tileflow.softmax_row
// CHECK: return
func.func @erase_identity_region(%x: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = tileflow.schedule_region #tileflow.schedule<tiles = [8, 8],
      order = [0, 1], unroll = 1, promote = [], target = "cpu"> {
    %1 = tileflow.softmax_row %x : tensor<8x8xf32>
    tileflow.yield %1 : tensor<8x8xf32>
  } : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

//===----------------------------------------------------------------------===//
// 3. Nested schedule regions collapse into one composed schedule.
//    min([6, 6], [4, 4]) = [4, 4], unroll 1 * 2 = 2, promote union = ["A"].
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func.func @collapse_nested_regions
// CHECK-COUNT-1: tileflow.schedule_region
// CHECK-SAME: tiles = [4, 4], order = [0, 1], unroll = 2, promote = ["A"]
// CHECK-NOT: tileflow.schedule_region
func.func @collapse_nested_regions(%x: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = tileflow.schedule_region #tileflow.schedule<tiles = [6, 6],
      order = [0, 1], unroll = 1, promote = [], target = "cpu"> {
    %1 = tileflow.schedule_region #tileflow.schedule<tiles = [4, 4],
        order = [0, 1], unroll = 2, promote = ["A"], target = "cpu"> {
      %2 = tileflow.softmax_row %x : tensor<8x8xf32>
      tileflow.yield %2 : tensor<8x8xf32>
    } : tensor<8x8xf32>
    tileflow.yield %1 : tensor<8x8xf32>
  } : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

//===----------------------------------------------------------------------===//
// 4. scale = 1.0 is removed from attention (the canonical default).
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func.func @attention_unit_scale
// CHECK: tileflow.attention
// CHECK-NOT: scale =
// CHECK: return
func.func @attention_unit_scale(%q: tensor<2x64x32xf32>, %k: tensor<2x64x32xf32>,
                                %v: tensor<2x64x32xf32>)
    -> tensor<2x64x32xf32> {
  %0 = tileflow.attention %q, %k, %v {scale = 1.0}
      : tensor<2x64x32xf32>, tensor<2x64x32xf32>, tensor<2x64x32xf32>
      -> tensor<2x64x32xf32>
  return %0 : tensor<2x64x32xf32>
}

//===----------------------------------------------------------------------===//
// 5. A matmul with a zero-sized dimension folds to an empty constant.
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func.func @zero_dim_matmul
// CHECK: tensor.empty() : tensor<0x16xf32>
// CHECK-NOT: tileflow.matmul
func.func @zero_dim_matmul(%a: tensor<0x8xf32>, %b: tensor<8x16xf32>)
    -> tensor<0x16xf32> {
  %0 = tileflow.matmul %a, %b : tensor<0x8xf32>, tensor<8x16xf32>
      -> tensor<0x16xf32>
  return %0 : tensor<0x16xf32>
}
