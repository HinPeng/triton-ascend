// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/MemoryModels.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <utility>
namespace mlir::triton::cfg {
namespace {
Type scalarType(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return tensor.getElementType();
  return type;
}
} // namespace
bool isBatchBoolToByteCast(Operation *op) {
  auto cast = dyn_cast_or_null<triton::BitcastOp>(op);
  if (!cast)
    return false;
  auto source =
      dyn_cast<triton::PointerType>(scalarType(cast.getSrc().getType()));
  auto result = dyn_cast<triton::PointerType>(scalarType(cast.getType()));
  return source && result && source.getAddressSpace() == 1 &&
         result.getAddressSpace() == 1 &&
         source.getPointeeType().isInteger(1) &&
         result.getPointeeType().isInteger(8);
}
bool isBatchNormalizedBoolLoad(Operation *op) {
  auto load = dyn_cast_or_null<triton::LoadOp>(op);
  if (!load)
    return false;
  auto marker = load->getAttrOfType<BoolAttr>("was_bool_to_int8");
  return marker && marker.getValue() &&
         isBatchBoolToByteCast(load.getPtr().getDefiningOp());
}
namespace {
unsigned zeroBits(Value value, unsigned depth = 0) {
  if (depth > 32)
    return 0;
  Operation *op = value.getDefiningOp();
  if (!op)
    return 0;
  if (auto c = dyn_cast<arith::ConstantOp>(op)) {
    if (auto integer = dyn_cast<IntegerAttr>(c.getValue()))
      return integer.getValue().countTrailingZeros();
    if (auto dense = dyn_cast<DenseIntElementsAttr>(c.getValue());
        dense && dense.isSplat())
      return dense.getSplatValue<APInt>().countTrailingZeros();
    return 0;
  }
  if (auto splat = dyn_cast<triton::SplatOp>(op))
    return zeroBits(splat.getSrc(), depth + 1);
  auto name = op->getName().getStringRef();
  if (name == "arith.muli")
    return std::min(64u, zeroBits(op->getOperand(0), depth + 1) +
                             zeroBits(op->getOperand(1), depth + 1));
  if (name == "arith.addi" || name == "arith.subi")
    return std::min(zeroBits(op->getOperand(0), depth + 1),
                    zeroBits(op->getOperand(1), depth + 1));
  if (name == "arith.extsi" || name == "arith.extui" ||
      name == "arith.trunci") {
    Type type = op->getResult(0).getType();
    if (auto tensor = dyn_cast<RankedTensorType>(type))
      type = tensor.getElementType();
    return std::min(cast<IntegerType>(type).getWidth(),
                    zeroBits(op->getOperand(0), depth + 1));
  }
  return 0;
}
std::optional<BatchPointerPlan> match(Value value, unsigned bytes,
                                      unsigned depth) {
  if (depth > 32)
    return std::nullopt;
  Operation *op = value.getDefiningOp();
  if (!op)
    return std::nullopt;
  if (auto cast = dyn_cast<triton::PtrToIntOp>(op))
    return BatchPointerPlan{cast.getSrc(), {}};
  if (auto splat = dyn_cast<triton::SplatOp>(op))
    return match(splat.getSrc(), bytes, depth + 1);
  auto name = op->getName().getStringRef();
  if (name != "arith.addi" && name != "arith.subi")
    return std::nullopt;
  auto lhs = match(op->getOperand(0), bytes, depth + 1);
  auto rhs = match(op->getOperand(1), bytes, depth + 1);
  if (static_cast<bool>(lhs) == static_cast<bool>(rhs) ||
      (rhs && name == "arith.subi"))
    return std::nullopt;
  bool left = static_cast<bool>(lhs);
  Value delta = op->getOperand(left ? 1 : 0);
  unsigned required = bytes == 8 ? 3 : bytes == 4 ? 2 : 1;
  if (zeroBits(delta) < required)
    return std::nullopt;
  auto result = left ? std::move(*lhs) : std::move(*rhs);
  result.byteOffsets.emplace_back(delta, name == "arith.subi");
  return result;
}
} // namespace
std::optional<BatchPointerPlan> getBatchPointerPlan(Value address,
                                                    unsigned bytes) {
  if (bytes != 2 && bytes != 4 && bytes != 8)
    return std::nullopt;
  return match(address, bytes, 0);
}
} // namespace mlir::triton::cfg
