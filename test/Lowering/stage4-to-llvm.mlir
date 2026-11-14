// RUN: tileflow-opt -tileflow-to-linalg -tileflow-apply-schedule='tile-sizes=16,16,16 order=0,1,2 fuse=true promote=true' -tileflow-to-loops -tileflow-to-llvm %s | FileCheck %s
// RUN: tileflow-opt -tileflow-to-linalg -tileflow-apply-schedule='tile-sizes=16,16,16 order=0,1,2 fuse=true promote=true' -tileflow-to-loops -tileflow-to-llvm %s | mlir-translate --mlir-to-llvmir | FileCheck %s --check-prefix=LLVMIR
//
// Stage 4: the module only contains LLVM dialect operations and translates to
// LLVM IR without diagnostics.

// CHECK-LABEL: llvm.func @matmul
// CHECK: llvm.mlir.constant
// CHECK: llvm.load
// CHECK: llvm.store
// CHECK: llvm.fmul
// CHECK: llvm.fadd
// CHECK: llvm.return
// CHECK-NOT: linalg.
// CHECK-NOT: scf.
// CHECK-NOT: memref.
// CHECK-NOT: affine.
// CHECK-NOT: arith.
// CHECK-NOT: func.func

// LLVMIR: define {{.*}} @matmul(
// LLVMIR: fmul float
// LLVMIR: fadd float
func.func @matmul(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>)
    -> tensor<64x64xf32> {
  %0 = tileflow.matmul %a, %b {
    schedule = #tileflow.schedule<tiles = [16, 16, 16], order = [0, 1, 2],
                                  unroll = 1, promote = ["A", "B"],
                                  target = "cpu">
  } : tensor<64x64xf32>, tensor<64x64xf32> -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}
