// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_ACCESS_ANALYSIS_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_ACCESS_ANALYSIS_H
#include "TritonToGraph/BlockBatching/BatchAccessFact.h"
#include "TritonToGraph/BlockBatching/BatchLoopFacts.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include <string>
namespace mlir::triton::cfg {
// Borrowed handles indexed by the original traversal. Invalidate before
// erasing the source body; the owned facts/projection survive that boundary.
struct BatchSourceIndex {
  SmallVector<Operation *> accesses;
  llvm::DenseMap<Operation *, unsigned> loopIDs;
  void clear() {
    accesses.clear();
    loopIDs.clear();
  }
};
struct BatchAccessReport {
  BatchSourceIndex sources;
  BatchEffectProjection effects;
  std::string guardJSON;
  std::string reason;
  Operation *operation = nullptr;
  SmallVector<BatchAccessFact> accesses;
  // 函数里的循环，每条一次，按 BatchLoopFact::id 升序。
  //
  // 和 BatchAccessFact::loops 装的是同一批事实，区别只在视角：那边按访存挂，
  // 同一条循环在每条访存下各出现一次；判定层问的是「这条程序轴循环安不安全」，
  // 要的是循环本身。
  //
  // 只收**包着某条访存**的循环。一条循环里一条访存都没有时这里没有它的事实，
  // 驱动层按证不出处理——那种形状合并了也没有收益，保守拒掉即可。这样做是为了
  // 不新增失败模式：界的表达式要过 Analysis::integer，而它证不出时会把整条
  // kernel 判拒，对一条本来就够不着的循环付这个代价不值得。
  SmallVector<BatchLoopFact, 0> loops;
};
unsigned getBatchIndexBitWidth(ModuleOp module);
BatchAccessReport analyzeBatchAccesses(triton::FuncOp function,
                                       unsigned indexBits,
                                       bool pointerRecurrences = false);
} // namespace mlir::triton::cfg
#endif
