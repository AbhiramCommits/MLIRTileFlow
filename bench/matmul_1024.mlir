// Benchmark and tuning input: a 1024x1024x1024 f32 matmul.
// `tileflow-tune` annotates this op with the schedule selected by the cost
// model; `tileflow-opt -tileflow-apply-schedule` consumes it.
func.func @matmul_1024(%a: tensor<1024x1024xf32>,
                       %b: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
  %0 = tileflow.matmul %a, %b
      : tensor<1024x1024xf32>, tensor<1024x1024xf32>
      -> tensor<1024x1024xf32>
  return %0 : tensor<1024x1024xf32>
}
