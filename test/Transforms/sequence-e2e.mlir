// RUN: tileflow-opt -tileflow-to-linalg -tileflow-tile-for-locality -tileflow-fuse-producers -tileflow-promote-to-scratchpad -tileflow-to-loops -tileflow-to-llvm %s | FileCheck %s
// RUN: tileflow-opt -tileflow-to-linalg -tileflow-tile-for-locality -tileflow-fuse-producers -tileflow-promote-to-scratchpad -tileflow-to-loops -tileflow-to-llvm %s | mlir-translate --mlir-to-llvmir | FileCheck %s --check-prefix=LLVMIR
//
// All three locality passes run in sequence on both a matmul and an attention
// function, then lower all the way to LLVM IR.

// CHECK-LABEL: llvm.func @matmul
// CHECK: llvm.load
// CHECK: llvm.fmul
// CHECK: llvm.fadd
// CHECK-LABEL: llvm.func @attn
// CHECK: llvm.load
// CHECK: llvm.fmul
// CHECK: llvm.fadd
// CHECK: llvm.store

// LLVMIR: define {{.*}} @matmul(
// LLVMIR: define {{.*}} @attn(

func.func @matmul(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>)
    -> tensor<64x64xf32> {
  %0 = tileflow.matmul %a, %b {
    schedule = #tileflow.schedule<tiles = [16, 16, 16], order = [0, 1, 2],
                                  unroll = 1, promote = ["A", "B"],
                                  target = "cpu">
  } : tensor<64x64xf32>, tensor<64x64xf32> -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

func.func @attn(%q: tensor<1x32x32xf32>, %k: tensor<1x32x32xf32>,
                %v: tensor<1x32x32xf32>) -> tensor<1x32x32xf32> {
  %0 = tileflow.attention %q, %k, %v {
    schedule = #tileflow.schedule<tiles = [1, 16, 16, 32], order = [0, 1, 2, 3],
                                  unroll = 1, promote = ["K", "V"],
                                  target = "cpu">
  } : tensor<1x32x32xf32>, tensor<1x32x32xf32>, tensor<1x32x32xf32>
      -> tensor<1x32x32xf32>
  return %0 : tensor<1x32x32xf32>
}
