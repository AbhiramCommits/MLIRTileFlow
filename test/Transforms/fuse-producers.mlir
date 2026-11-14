// RUN: tileflow-opt -tileflow-to-linalg -tileflow-tile-for-locality -tileflow-fuse-producers -tileflow-to-loops %s | FileCheck %s
//
// Fusing the elementwise producer into the tiled matmul loop keeps the
// activation temporary at tile size. The only full-size allocation left is
// the 64x128 matmul result; the 64x64 `%scaled` temporary never materializes.

// CHECK-LABEL: func.func @fused
// CHECK: memref.alloc() {{.*}}memref<64x128xf32>
// CHECK-NOT: memref.alloc() {{.*}}memref<64x64xf32>
// CHECK: arith.mulf
// CHECK: memref.load
// CHECK: memref.store
// CHECK-NOT: tensor<
func.func @fused(%a: tensor<64x64xf32>, %b: tensor<64x128xf32>)
    -> tensor<64x128xf32> {
  %c2 = arith.constant 2.0 : f32
  %empty = tensor.empty() : tensor<64x64xf32>
  %scaled = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%a : tensor<64x64xf32>) outs(%empty : tensor<64x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %m = arith.mulf %in, %c2 : f32
    linalg.yield %m : f32
  } -> tensor<64x64xf32>
  %0 = tileflow.matmul %scaled, %b {
    schedule = #tileflow.schedule<tiles = [16, 16, 16], order = [0, 1, 2],
                                  unroll = 1, promote = [], target = "cpu">
  } : tensor<64x64xf32>, tensor<64x128xf32> -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}
