// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_ACCESS_FACT_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_ACCESS_FACT_H
#include "TritonToGraph/BlockBatching/BatchLoopFacts.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::cfg {
// Owned original-domain facts; no IR handles or serialized representation.
struct BatchAccessFact {
  int root = -1;
  unsigned bytes = 0;
  bool write = false;
  llvm::SmallVector<int64_t> shape;
  BatchExprRef byteOffset, mask;
  llvm::SmallVector<BatchExprRef> pointerChecks;
  // 包着这条访存的 scf.for，由内向外。S2b 之前这里直接存 JSON，判定层要用就
  // 得先把字符串解回来；现在存事实本身，序列化留给报告路径（BatchFactsJSON.h）。
  // 零内联：绝大多数访存根本不在循环里，有循环时再上堆。默认内联会让
  // BatchAccessFact 直接撑过 SmallVector 的 256 字节门槛。
  llvm::SmallVector<BatchLoopFact, 0> loops;
  bool isVolatile = false;
  // Missing summaries retain a real device SSA effect, never an empty domain.
  bool deviceAddress = false;
  bool normalizedPointer = false;
};

// Guard effects are a projection, not the original accesses used by layout.
// Member slots and pointer checks survive joint-store normalization.
struct BatchEffectProjection {
  llvm::SmallVector<BatchAccessFact> accesses;
};
} // namespace mlir::triton::cfg
#endif
