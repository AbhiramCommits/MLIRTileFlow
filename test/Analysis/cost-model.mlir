// RUN: python3 %S/cost_model_check.py tileflow-opt %s
//
// The cost model must rank an untiled matmul as strictly more DRAM traffic
// than a cache-resident tiled one, and reject a schedule whose promoted tile
// footprint exceeds the scratchpad.

func.func @untiled(%a: tensor<1024x1024xf32>, %b: tensor<1024x1024xf32>)
    -> tensor<1024x1024xf32> {
  %0 = tileflow.matmul %a, %b {
    schedule = #tileflow.schedule<tiles = [1024, 1024, 1024], order = [0, 1, 2],
                                  unroll = 1, promote = [], target = "cpu">
  } : tensor<1024x1024xf32>, tensor<1024x1024xf32> -> tensor<1024x1024xf32>
  return %0 : tensor<1024x1024xf32>
}

func.func @tiled(%a: tensor<1024x1024xf32>, %b: tensor<1024x1024xf32>)
    -> tensor<1024x1024xf32> {
  %0 = tileflow.matmul %a, %b {
    schedule = #tileflow.schedule<tiles = [32, 32, 32], order = [0, 1, 2],
                                  unroll = 1, promote = [], target = "cpu">
  } : tensor<1024x1024xf32>, tensor<1024x1024xf32> -> tensor<1024x1024xf32>
  return %0 : tensor<1024x1024xf32>
}

func.func @over_capacity(%a: tensor<1024x1024xf32>, %b: tensor<1024x1024xf32>)
    -> tensor<1024x1024xf32> {
  %0 = tileflow.matmul %a, %b {
    schedule = #tileflow.schedule<tiles = [512, 512, 512], order = [0, 1, 2],
                                  unroll = 1, promote = ["A", "B"],
                                  target = "cpu">
  } : tensor<1024x1024xf32>, tensor<1024x1024xf32> -> tensor<1024x1024xf32>
  return %0 : tensor<1024x1024xf32>
}
