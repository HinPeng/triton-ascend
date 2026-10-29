// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_CONTROL_FLOW_MODELS_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_CONTROL_FLOW_MODELS_H
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include <string>
namespace mlir::triton::cfg {
struct BatchControlFlowPlan {
  llvm::DenseMap<Value, bool> varying;
  std::string reason;
  Operation *operation = nullptr;
  bool isVarying(Value value) const { return varying.lookup(value); }
};
// Loads may be U only under the separate whole-entry memory noninterference
// proof. This plan never authorizes a launch without that runtime guard.
BatchControlFlowPlan analyzeBatchControlFlow(triton::FuncOp function, int axis);
} // namespace mlir::triton::cfg
#endif
