// RUN: tileflow-opt -tileflow-remarks -tileflow-to-linalg -tileflow-tile-for-locality -tileflow-fuse-producers %s 2>&1 | FileCheck %s --check-prefix=REMARK
//
// -tileflow-remarks makes each transformation describe what it did.

// REMARK: remark: tileflow tiled op with [16, 16, 16]
func.func @matmul(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>)
    -> tensor<64x64xf32> {
  %0 = tileflow.matmul %a, %b {
    schedule = #tileflow.schedule<tiles = [16, 16, 16], order = [0, 1, 2],
                                  unroll = 1, promote = [], target = "cpu">
  } : tensor<64x64xf32>, tensor<64x64xf32> -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}
