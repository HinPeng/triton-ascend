// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/ReductionModels.h"
#include "TritonToGraph/BlockBatching/BatchOpRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::triton::cfg {
namespace {
bool scalarNumerical(Type type) {
  return type.isF16() || type.isBF16() || type.isF32() || type.isInteger(1) ||
         type.isInteger(32) || type.isInteger(64);
}
} // namespace
LogicalResult verifyBatchReduction(Operation *op, std::string &reason,
                                   Operation *&failedOp) {
  auto reject = [&](Operation *at, llvm::StringRef why) {
    failedOp = at;
    reason = why.str();
    return failure();
  };
  if (op->getNumRegions() != 1 || !op->getRegion(0).hasOneBlock() ||
      !op->getNumOperands() || op->getNumResults() != op->getNumOperands())
    return reject(op, "UNSUPPORTED_REDUCTION");
  auto source = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto axis = op->getAttrOfType<IntegerAttr>("axis");
  if (!source || !axis || axis.getInt() < 0 ||
      axis.getInt() >= source.getRank())
    return reject(op, "UNSUPPORTED_REDUCTION");
  for (Value value : op->getOperands()) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type || type.getShape() != source.getShape() || type.getEncoding())
      return reject(op, "UNSUPPORTED_REDUCTION_TYPE");
    Type element = type.getElementType();
    if (!(element.isF32() || element.isInteger(32) || element.isInteger(64)))
      return reject(op, "UNSUPPORTED_REDUCTION_TYPE");
  }
  auto &body = op->getRegion(0).front();
  if (body.getNumArguments() != 2 * op->getNumOperands() ||
      body.getOperations().size() > 128)
    return reject(op, "UNSUPPORTED_REDUCTION_COMBINER");
  auto expectedReturn =
      isa<triton::ReduceOp>(op) ? "tt.reduce.return" : "tt.scan.return";
  if (body.empty() || body.back().getName().getStringRef() != expectedReturn ||
      body.back().getNumOperands() != op->getNumResults())
    return reject(op, "UNSUPPORTED_REDUCTION_COMBINER");
  // These tuple-return forms depended on the removed seeded converter.
  // Do not change the baseline converter to make a batching candidate work.
  if (isa<triton::ReduceOp>(op) && op->getNumResults() > 1) {
    for (Value value : body.back().getOperands()) {
      bool internal = false;
      if (auto argument = dyn_cast<BlockArgument>(value))
        internal = argument.getOwner() == &body;
      else
        internal = value.getDefiningOp()->getBlock() == &body;
      if (!internal)
        return reject(&body.back(), "UNSUPPORTED_REDUCTION_RETURN_CAPTURE");
    }
  }
  for (Operation &nested : body) {
    if (&nested == &body.back())
      continue;
    if (isa<triton::ReduceOp>(op)) {
      auto cmp = dyn_cast<arith::CmpFOp>(nested);
      if (cmp && cmp.getPredicate() == arith::CmpFPredicate::UNO)
        return reject(&nested, "UNSUPPORTED_NAN_REDUCTION_COMBINER");
    }
    auto name = nested.getName().getStringRef();
    auto *model = lookupBatchOp(name);
    if (!model || !model->implemented || nested.getNumRegions() ||
        failed(verifyBatchAttributes(&nested, *model)))
      return reject(&nested, "UNSUPPORTED_REDUCTION_COMBINER");
    // An IEEE elementwise policy is not a declaration that the operation is
    // an accepted reduction/scan combiner. Keep that subset unchanged.
    if (model->inactiveInputs == BatchInactivePolicy::IEEE)
      return reject(&nested, "UNSUPPORTED_REDUCTION_COMBINER");
    bool constant = model->model == BatchModel::Constant;
    bool totalArithmetic =
        model->model == BatchModel::Elementwise && name.starts_with("arith.") &&
        model->inactiveInputs == BatchInactivePolicy::Preserve;
    if (!constant && !totalArithmetic)
      return reject(&nested, "INACTIVE_COMBINER_UNSAFE");
    for (Type type : nested.getOperandTypes())
      if (!scalarNumerical(type))
        return reject(&nested, "UNSUPPORTED_COMBINER_TYPE");
    for (Type type : nested.getResultTypes())
      if (!scalarNumerical(type))
        return reject(&nested, "UNSUPPORTED_COMBINER_TYPE");
  }
  return success();
}

SmallVector<Value> getBatchCombinerCaptures(Operation *op) {
  Region *region = &op->getRegion(0);
  SmallVector<Value> values;
  llvm::SmallPtrSet<void *, 16> seen;
  for (Operation &nested : region->front())
    for (Value value : nested.getOperands()) {
      bool internal = false;
      if (auto arg = dyn_cast<BlockArgument>(value))
        internal = arg.getOwner()->getParent() == region;
      else
        internal = value.getDefiningOp()->getParentRegion() == region;
      if (!internal && seen.insert(value.getAsOpaquePointer()).second)
        values.push_back(value);
    }
  return values;
}

Operation *rewriteBatchReduction(Operation *op, ValueRange inputs,
                                 TypeRange results, IRMapping &mapping,
                                 IRRewriter &rewriter) {
  IRMapping operands = mapping;
  for (auto [original, value] : llvm::zip(op->getOperands(), inputs))
    operands.map(original, value);
  Operation *result = rewriter.clone(*op, operands);
  result->setAttr("axis",
                  rewriter.getI32IntegerAttr(
                      op->getAttrOfType<IntegerAttr>("axis").getInt() + 1));
  for (auto [value, type] : llvm::zip(result->getResults(), results))
    value.setType(type);
  return result;
}
} // namespace mlir::triton::cfg
