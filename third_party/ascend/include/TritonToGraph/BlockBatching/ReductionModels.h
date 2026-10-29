// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_REDUCTION_MODELS_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_REDUCTION_MODELS_H
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include <string>
namespace mlir::triton::cfg {
LogicalResult verifyBatchReduction(Operation *op, std::string &reason,
                                   Operation *&failedOp);
SmallVector<Value> getBatchCombinerCaptures(Operation *op);
Operation *rewriteBatchReduction(Operation *op, ValueRange inputs,
                                 TypeRange results, IRMapping &mapping,
                                 IRRewriter &rewriter);
} // namespace mlir::triton::cfg
#endif
