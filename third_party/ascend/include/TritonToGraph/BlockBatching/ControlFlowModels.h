// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_CONTROL_FLOW_MODELS_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_CONTROL_FLOW_MODELS_H
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <string>
namespace mlir::triton::cfg {
struct BatchControlFlowPlan {
  llvm::DenseMap<Value, bool> varying;
  std::string reason;
  Operation *operation = nullptr;
  bool isVarying(Value value) const { return varying.lookup(value); }
};
// Uniform loads require a scoped ShareInvariantLoad noninterference proof.
// Opaque reads retain lane multiplicity; rewriting consumes this fixed point.
BatchControlFlowPlan analyzeBatchControlFlow(
    triton::FuncOp function, int axis,
    const llvm::DenseSet<Operation *> &replicatedLoads = {});
} // namespace mlir::triton::cfg
#endif
