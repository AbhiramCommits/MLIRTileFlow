// RUN: tileflow-opt -tileflow-lower='stop-after=linalg' %s | FileCheck %s --check-prefix=LINALG
// RUN: tileflow-opt -tileflow-lower='stop-after=schedule' %s | FileCheck %s --check-prefix=SCHEDULE
// RUN: tileflow-opt -tileflow-lower='stop-after=loops' %s | FileCheck %s --check-prefix=LOOPS
// RUN: tileflow-opt -tileflow-lower='stop-after=llvm' %s | FileCheck %s --check-prefix=LLVM
//
// The composite pipeline can be halted at each stage.

// LINALG-LABEL: func.func @matmul
// LINALG: tensor.empty() : tensor<32x32xf32>
// LINALG: linalg.matmul
// LINALG-NOT: scf.for

// SCHEDULE-LABEL: func.func @matmul
// SCHEDULE: memref<32x32xf32>
// SCHEDULE: scf.for
// SCHEDULE: linalg.matmul

// LOOPS-LABEL: func.func @matmul
// LOOPS: memref<32x32xf32>
// LOOPS: scf.for
// LOOPS: memref.load
// LOOPS-NOT: linalg.

// LLVM-LABEL: llvm.func @matmul
// LLVM: llvm.load
// LLVM-NOT: scf.
// LLVM-NOT: linalg.
// LLVM-NOT: memref.
func.func @matmul(%a: tensor<32x32xf32>, %b: tensor<32x32xf32>)
    -> tensor<32x32xf32> {
  %0 = tileflow.matmul %a, %b {
    schedule = #tileflow.schedule<tiles = [16, 16, 16], order = [0, 1, 2],
                                  unroll = 1, promote = ["A", "B"],
                                  target = "cpu">
  } : tensor<32x32xf32>, tensor<32x32xf32> -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}
