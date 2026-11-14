// RUN: tileflow-opt -verify-diagnostics %s
//
// Negative coverage for every verifier branch. Each operation is on a single
// line so `@below` unambiguously refers to it.

//===----------------------------------------------------------------------===//
// Matmul verifier
//===----------------------------------------------------------------------===//

func.func @matmul_k_mismatch(%a: tensor<4x8xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // expected-error @below {{K dimension mismatch: A has K=8 but B has K=4}}
  %0 = tileflow.matmul %a, %b : tensor<4x8xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

func.func @matmul_bad_result(%a: tensor<4x8xf32>, %b: tensor<8x4xf32>) -> tensor<4x8xf32> {
  // expected-error @below {{result shape [4, 8] does not match the expected 4x4}}
  %0 = tileflow.matmul %a, %b : tensor<4x8xf32>, tensor<8x4xf32> -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

func.func @matmul_bad_schedule_rank(%a: tensor<4x8xf32>, %b: tensor<8x4xf32>) -> tensor<4x4xf32> {
  // expected-error @below {{schedule has 2 tile sizes but matmul iteration space (M, N, K) has rank 3}}
  %0 = tileflow.matmul %a, %b {schedule = #tileflow.schedule<tiles = [4, 4], order = [0, 1], unroll = 1, promote = [], target = "cpu">} : tensor<4x8xf32>, tensor<8x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

//===----------------------------------------------------------------------===//
// Schedule attribute verifier
//===----------------------------------------------------------------------===//

func.func @schedule_zero_tile(%a: tensor<4x8xf32>, %b: tensor<8x4xf32>) -> tensor<4x4xf32> {
  // expected-error @below {{tileflow.schedule tile size at position 1 must be strictly positive, got 0}}
  %0 = tileflow.matmul %a, %b {schedule = #tileflow.schedule<tiles = [4, 0, 4], order = [0, 1, 2], unroll = 1, promote = [], target = "cpu">} : tensor<4x8xf32>, tensor<8x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

func.func @schedule_bad_order(%a: tensor<4x8xf32>, %b: tensor<8x4xf32>) -> tensor<4x4xf32> {
  // expected-error @below {{tileflow.schedule order is not a permutation: 1 appears more than once}}
  %0 = tileflow.matmul %a, %b {schedule = #tileflow.schedule<tiles = [4, 4, 4], order = [0, 1, 1], unroll = 1, promote = [], target = "cpu">} : tensor<4x8xf32>, tensor<8x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

func.func @schedule_bad_promote(%a: tensor<4x8xf32>, %b: tensor<8x4xf32>) -> tensor<4x4xf32> {
  // expected-error @below {{tileflow.schedule promotes unknown operand 'C'; expected a subset of {"A", "B", "Q", "K", "V"}}}
  %0 = tileflow.matmul %a, %b {schedule = #tileflow.schedule<tiles = [4, 4, 4], order = [0, 1, 2], unroll = 1, promote = ["C"], target = "cpu">} : tensor<4x8xf32>, tensor<8x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

func.func @schedule_bad_target(%a: tensor<4x8xf32>, %b: tensor<8x4xf32>) -> tensor<4x4xf32> {
  // expected-error @below {{tileflow.schedule target must be "cpu" or "gpu", got "tpu"}}
  %0 = tileflow.matmul %a, %b {schedule = #tileflow.schedule<tiles = [4, 4, 4], order = [0, 1, 2], unroll = 1, promote = [], target = "tpu">} : tensor<4x8xf32>, tensor<8x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

func.func @schedule_zero_unroll(%a: tensor<4x8xf32>, %b: tensor<8x4xf32>) -> tensor<4x4xf32> {
  // expected-error @below {{tileflow.schedule unroll must be at least 1, got 0}}
  %0 = tileflow.matmul %a, %b {schedule = #tileflow.schedule<tiles = [4, 4, 4], order = [0, 1, 2], unroll = 0, promote = [], target = "cpu">} : tensor<4x8xf32>, tensor<8x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

//===----------------------------------------------------------------------===//
// Softmax verifier
//===----------------------------------------------------------------------===//

func.func @softmax_bad_axis(%x: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @below {{axis must be 0 or 1 for a rank-2 input, got 2}}
  %0 = tileflow.softmax_row %x {axis = 2} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

func.func @softmax_bad_rank(%x: tensor<8x8x8xf32>) -> tensor<8x8x8xf32> {
  // expected-error @below {{expects a rank-2 tensor, got rank 3}}
  %0 = tileflow.softmax_row %x : tensor<8x8x8xf32>
  return %0 : tensor<8x8x8xf32>
}

//===----------------------------------------------------------------------===//
// Attention verifier
//===----------------------------------------------------------------------===//

func.func @attention_head_mismatch(%q: tensor<2x64x32xf32>, %k: tensor<4x64x32xf32>, %v: tensor<2x64x32xf32>) -> tensor<2x64x32xf32> {
  // expected-error @below {{attention head count mismatch: Q has 2 heads but K has 4}}
  %0 = tileflow.attention %q, %k, %v : tensor<2x64x32xf32>, tensor<4x64x32xf32>, tensor<2x64x32xf32> -> tensor<2x64x32xf32>
  return %0 : tensor<2x64x32xf32>
}

func.func @attention_dim_mismatch(%q: tensor<2x64x32xf32>, %k: tensor<2x64x16xf32>, %v: tensor<2x64x32xf32>) -> tensor<2x64x32xf32> {
  // expected-error @below {{attention head dimension mismatch: Q has 32 but K has 16}}
  %0 = tileflow.attention %q, %k, %v : tensor<2x64x32xf32>, tensor<2x64x16xf32>, tensor<2x64x32xf32> -> tensor<2x64x32xf32>
  return %0 : tensor<2x64x32xf32>
}

func.func @attention_seq_mismatch(%q: tensor<2x64x32xf32>, %k: tensor<2x64x32xf32>, %v: tensor<2x32x32xf32>) -> tensor<2x64x32xf32> {
  // expected-error @below {{attention key sequence length mismatch: K has 64 keys but V has 32}}
  %0 = tileflow.attention %q, %k, %v : tensor<2x64x32xf32>, tensor<2x64x32xf32>, tensor<2x32x32xf32> -> tensor<2x64x32xf32>
  return %0 : tensor<2x64x32xf32>
}

func.func @attention_bad_causal(%q: tensor<2x64x32xf32>, %k: tensor<2x32x32xf32>, %v: tensor<2x32x32xf32>) -> tensor<2x64x32xf32> {
  // expected-error @below {{causal attention requires seq_q == seq_k, got seq_q=64 and seq_k=32}}
  %0 = tileflow.attention %q, %k, %v {causal} : tensor<2x64x32xf32>, tensor<2x32x32xf32>, tensor<2x32x32xf32> -> tensor<2x64x32xf32>
  return %0 : tensor<2x64x32xf32>
}

func.func @attention_bad_result(%q: tensor<2x64x32xf32>, %k: tensor<2x64x32xf32>, %v: tensor<2x64x32xf32>) -> tensor<2x64x16xf32> {
  // expected-error @below {{result shape [2, 64, 16] does not match [heads=2, seq_q=64, v_dim=32]}}
  %0 = tileflow.attention %q, %k, %v : tensor<2x64x32xf32>, tensor<2x64x32xf32>, tensor<2x64x32xf32> -> tensor<2x64x16xf32>
  return %0 : tensor<2x64x16xf32>
}

//===----------------------------------------------------------------------===//
// Schedule region verifier
//===----------------------------------------------------------------------===//

func.func @region_yield_type_mismatch(%x: tensor<8x8xf32>) -> tensor<4x4xf32> {
  // expected-error @below {{yielded value #0 has type 'tensor<8x8xf32>' but result #0 has type 'tensor<4x4xf32>'}}
  %0 = tileflow.schedule_region #tileflow.schedule<tiles = [4, 4], order = [0, 1], unroll = 1, promote = [], target = "cpu"> { %1 = tileflow.softmax_row %x : tensor<8x8xf32>
    tileflow.yield %1 : tensor<8x8xf32>
  } : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

func.func @region_bad_schedule_rank(%x: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @below {{schedule has 3 tile sizes but the region results have rank 2}}
  %0 = tileflow.schedule_region #tileflow.schedule<tiles = [4, 4, 4], order = [0, 1, 2], unroll = 1, promote = [], target = "cpu"> { %1 = tileflow.softmax_row %x : tensor<8x8xf32>
    tileflow.yield %1 : tensor<8x8xf32>
  } : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

func.func @matmul_unranked(%a: tensor<*xf32>, %b: tensor<*xf32>) -> tensor<*xf32> {
  // expected-error @below {{expects ranked tensor operands and result}}
  %0 = tileflow.matmul %a, %b : tensor<*xf32>, tensor<*xf32> -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func.func @matmul_bad_rank(%a: tensor<4x4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // expected-error @below {{expects rank-2 operands and result, got ranks (3, 2, 2)}}
  %0 = tileflow.matmul %a, %b : tensor<4x4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

func.func @attention_bad_rank(%q: tensor<64x32xf32>, %k: tensor<2x64x32xf32>, %v: tensor<2x64x32xf32>) -> tensor<2x64x32xf32> {
  // expected-error @below {{expects rank-3 [heads, seq, head_dim] tensors, got ranks (2, 3, 3, 3)}}
  %0 = tileflow.attention %q, %k, %v : tensor<64x32xf32>, tensor<2x64x32xf32>, tensor<2x64x32xf32> -> tensor<2x64x32xf32>
  return %0 : tensor<2x64x32xf32>
}

func.func @region_yield_count(%x: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @below {{yields 2 values but the region has 1 results}}
  %0 = tileflow.schedule_region #tileflow.schedule<tiles = [4, 4], order = [0, 1], unroll = 1, promote = [], target = "cpu"> { %1 = tileflow.softmax_row %x : tensor<8x8xf32>
    tileflow.yield %1, %1 : tensor<8x8xf32>, tensor<8x8xf32>
  } : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
