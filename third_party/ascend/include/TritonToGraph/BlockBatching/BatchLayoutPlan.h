// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_LAYOUT_PLAN_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_LAYOUT_PLAN_H
#include "TritonToGraph/BlockBatching/BatchAccessAnalysis.h"
namespace mlir::triton::cfg {
inline constexpr unsigned kBatchLayoutPlannerVersion = 2;
struct BatchLayoutTerm {
  BatchExpr::Kind kind;
  int64_t id;
  BatchExprRef coefficient;
  int64_t divisor = 0;
};
struct BatchLayoutPlan {
  enum Action { Keep, MemoryView, PointwiseRegion };
  Action action = Keep;
  unsigned accessID = 0;
  int gridAxis = -1;
  SmallVector<int64_t> localAxes, shape;
  int64_t tileExtent = 0, origin = 0;
  // Uniform integer expressions, independent of all program/local coordinates.
  // Owned DAGs survive the original IR, unlike an MLIR Value captured here.
  BatchExprRef elementOrigin, elementStride;
  bool symbolic = false;
  bool linearMemoryView = false;
  SmallVector<BatchLayoutTerm> restTerms;
  SmallVector<BatchExprRef> residualMasks;
  std::string reason = "DIRECTION_UNPROVEN", maskClass = "unknown";
  std::string location;
  BatchExprRef coordinate, upper;
  bool controlFlowIndependent = false;
};
// An owned outcome, never an IR handle. Layout attempts may downgrade this
// copy to Keep without changing the accepted source plan.
struct BatchLayoutResult {
  BatchLayoutPlan layout;
  bool rewritten = false;
  std::string runtimeGuardJSON;
};
SmallVector<BatchLayoutResult, 0>
initialBatchLayoutResults(ArrayRef<BatchLayoutPlan> plans);
std::string serializeBatchLayoutResults(ArrayRef<BatchLayoutResult> results);

SmallVector<BatchLayoutPlan, 0>
planBatchLayouts(const BatchAccessReport &accesses, int axis, int64_t factor);
// Read-only proof for automatic axis selection. Uses the same closed-region
// check as the memory rewrite and requires unit-stride column accesses.
bool hasBatchColumnPointwiseRegion(triton::FuncOp function,
                                   const BatchAccessReport &accesses, int axis,
                                   int64_t factor);

std::string serializeBatchLayouts(ArrayRef<BatchLayoutPlan> plans);
LogicalResult
rewriteBatchMemoryViews(ModuleOp module, const BatchAccessReport &accesses,
                        ArrayRef<BatchLayoutPlan> plans,
                        SmallVectorImpl<BatchLayoutResult> &results,
                        ArrayRef<Operation *> semanticAccesses, Value group,
                        unsigned originalArgCount, int axis, int64_t factor);
} // namespace mlir::triton::cfg
#endif
