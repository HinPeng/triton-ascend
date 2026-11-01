// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_ACCESS_ANALYSIS_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_ACCESS_ANALYSIS_H
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/JSON.h"
#include <memory>
#include <string>
namespace mlir::triton::cfg {
// Owned, immutable coordinate expressions. Coordinate IDs denote projections
// in this access, not make_range SSA identity. No erased SSA value is retained.
struct BatchExpr;
using BatchExprRef = std::shared_ptr<const BatchExpr>;
struct BatchExpr {
  enum Kind {
    Constant,
    Argument,
    Program,
    Grid,
    Coordinate,
    Induction,
    Add,
    Sub,
    Mul,
    ExtSI,
    ExtUI,
    Trunc,
    True,
    False,
    Unknown,
    And,
    SLT,
    ULT,
    EQ,
    MinSI,
    MaxSI
  };
  Kind kind;
  unsigned bits = 0;
  int64_t value = 0, extent = 0;
  BatchExprRef lhs, rhs;
  bool rangeOrigin = false;
  static BatchExprRef range(BatchExprRef start, BatchExprRef projection);
  static BatchExprRef leaf(Kind kind, int64_t value = 0, unsigned bits = 0,
                           int64_t extent = 0);
  static BatchExprRef node(Kind kind, unsigned bits, BatchExprRef lhs,
                           BatchExprRef rhs = {}, unsigned sourceBits = 0);
  llvm::json::Value json() const;
};
llvm::json::Value batchExprJSON(const BatchExprRef &expr);
struct BatchAccessFact {
  // Valid only until the original body is erased. Consumers must copy plans
  // and map this operation while both original and generated bodies are alive.
  Operation *operation = nullptr;
  int root = -1;
  unsigned bytes = 0;
  bool write = false;
  SmallVector<int64_t> shape;
  BatchExprRef byteOffset, mask;
  SmallVector<BatchExprRef> pointerChecks;
  llvm::json::Array loops;
  bool isVolatile = false;
  // Missing summaries retain a real device SSA effect, never an empty domain.
  bool deviceAddress = false;
  bool normalizedPointer = false;
};
struct BatchAccessReport {
  std::string guardJSON;
  std::string reason;
  Operation *operation = nullptr;
  SmallVector<BatchAccessFact> accesses;
};
unsigned getBatchIndexBitWidth(ModuleOp module);
BatchAccessReport analyzeBatchAccesses(triton::FuncOp function,
                                       unsigned indexBits);
} // namespace mlir::triton::cfg
#endif
