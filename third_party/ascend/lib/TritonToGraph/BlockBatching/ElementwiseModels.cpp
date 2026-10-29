// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/ElementwiseModels.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::triton::cfg {
bool needsBatchActiveMask(const BatchOpDescriptor &descriptor) {
  return descriptor.inactiveInputs != BatchInactivePolicy::Preserve &&
         descriptor.inactiveInputs != BatchInactivePolicy::IEEE;
}

SmallVector<Value>
prepareBatchElementwiseOperands(const BatchOpDescriptor &descriptor,
                                ValueRange operands, Value activeMask,
                                Location loc, IRRewriter &rewriter) {
  SmallVector<Value> result(operands.begin(), operands.end());
  if (!needsBatchActiveMask(descriptor))
    return result;
  assert(activeMask && "poison-sensitive operands require a live mask");
  for (auto [index, operand] : llvm::enumerate(operands)) {
    int64_t fill = 0;
    if (descriptor.inactiveInputs == BatchInactivePolicy::Positive ||
        (descriptor.inactiveInputs == BatchInactivePolicy::Divide &&
         index == 1))
      fill = 1;
    auto type = cast<RankedTensorType>(operand.getType());
    Type element = type.getElementType();
    TypedAttr value;
    if (isa<IntegerType>(element))
      value = rewriter.getIntegerAttr(element, fill);
    else
      value = rewriter.getFloatAttr(element, static_cast<double>(fill));
    Value safe = rewriter.create<arith::ConstantOp>(
        loc, type, DenseElementsAttr::get(type, value));
    // A select of the RESULT would be too late for integer division, shifts,
    // or float-to-int conversion: sanitize the operands before evaluation.
    result[index] =
        rewriter.create<arith::SelectOp>(loc, activeMask, operand, safe);
  }
  return result;
}
Operation *rewriteBatchElementwise(Operation *op,
                                   const BatchOpDescriptor &descriptor,
                                   ValueRange operands,
                                   RankedTensorType resultType, Location loc,
                                   IRRewriter &rewriter) {
  if (descriptor.name == "arith.ceildivui") {
    // q + (r != 0) is exact throughout the original unsigned domain. It
    // cannot overflow: b == 1 implies r == 0; b > 1 leaves room for +1.
    Value quotient =
        rewriter.create<arith::DivUIOp>(loc, operands[0], operands[1]);
    Value remainder =
        rewriter.create<arith::RemUIOp>(loc, operands[0], operands[1]);
    TypedAttr zero = rewriter.getIntegerAttr(resultType.getElementType(), 0);
    Value zeros = rewriter.create<arith::ConstantOp>(
        loc, resultType, DenseElementsAttr::get(resultType, zero));
    Value nonzero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne, remainder, zeros);
    Value increment = rewriter.create<arith::ExtUIOp>(loc, resultType, nonzero);
    return rewriter.create<arith::AddIOp>(loc, quotient, increment)
        .getOperation();
  }
  IRMapping mapping;
  for (auto [original, value] : llvm::zip(op->getOperands(), operands))
    mapping.map(original, value);
  Operation *result = rewriter.clone(*op, mapping);
  result->getResult(0).setType(resultType);
  return result;
}
} // namespace mlir::triton::cfg
