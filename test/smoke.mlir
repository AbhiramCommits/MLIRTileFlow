// RUN: tileflow-opt %s | tileflow-opt | FileCheck %s
// RUN: tileflow-opt --show-dialects | FileCheck %s --check-prefix=DIALECT

// CHECK: func.func @smoke
// DIALECT: tileflow

func.func @smoke(%a: tensor<4xf32>) -> tensor<4xf32> {
  %0 = arith.negf %a : tensor<4xf32>
  return %0 : tensor<4xf32>
}
