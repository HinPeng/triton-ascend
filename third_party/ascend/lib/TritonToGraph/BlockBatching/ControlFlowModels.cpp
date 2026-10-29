// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/ControlFlowModels.h"
#include "TritonToGraph/BlockBatching/ReductionModels.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"
namespace mlir::triton::cfg {
namespace {
struct Analysis {
  BatchControlFlowPlan plan;
  int axis;
  unsigned budget = 8192;
  explicit Analysis(int axis) : axis(axis) {}
  bool reject(Operation *op, llvm::StringRef reason) {
    if (plan.reason.empty()) {
      plan.reason = reason.str();
      plan.operation = op;
    }
    return false;
  }
  bool known(Value value, Operation *op) {
    return plan.varying.count(value) || reject(op, "VALUE_VARIATION_UNKNOWN");
  }
  void record(Value value, bool varying) {
    plan.varying[value] = plan.isVarying(value) || varying;
  }
  bool block(Block &body, unsigned depth = 0) {
    if (depth > 8)
      return reject(body.getParentOp(), "REGION_DEPTH_EXCEEDED");
    for (Operation &op : body) {
      if (!budget--)
        return reject(&op, "ANALYSIS_BUDGET_EXCEEDED");
      for (Value value : op.getOperands())
        if (!known(value, &op))
          return false;
      if (auto branch = dyn_cast<scf::IfOp>(op)) {
        if (plan.isVarying(branch.getCondition()))
          return reject(&op, "VARYING_CONTROL_CONDITION");
        if (!branch.getThenRegion().hasOneBlock() ||
            (!branch.getElseRegion().empty() &&
             !branch.getElseRegion().hasOneBlock()))
          return reject(&op, "UNSUPPORTED_REGION_SHAPE");
        if (!block(branch.getThenRegion().front(), depth + 1))
          return false;
        if (!branch.getElseRegion().empty() &&
            !block(branch.getElseRegion().front(), depth + 1))
          return false;
        if (branch.getNumResults()) {
          auto thenYield =
              cast<scf::YieldOp>(branch.getThenRegion().front().back());
          auto elseYield =
              cast<scf::YieldOp>(branch.getElseRegion().front().back());
          for (auto [result, a, b] :
               llvm::zip(branch.getResults(), thenYield.getOperands(),
                         elseYield.getOperands()))
            record(result, plan.isVarying(a) || plan.isVarying(b));
        }
        continue;
      }
      if (auto loop = dyn_cast<scf::ForOp>(op)) {
        if (plan.isVarying(loop.getLowerBound()) ||
            plan.isVarying(loop.getUpperBound()) ||
            plan.isVarying(loop.getStep()))
          return reject(&op, "VARYING_LOOP_BOUND");
        auto step = loop.getStep().getDefiningOp<arith::ConstantOp>();
        auto amount =
            step ? dyn_cast<IntegerAttr>(step.getValue()) : IntegerAttr{};
        if (!amount || amount.getValue().isNegative() ||
            amount.getValue().isZero())
          return reject(&op, "UNPROVEN_POSITIVE_LOOP_STEP");
        if (loop.getNumRegionIterArgs() > 128)
          return reject(&op, "ANALYSIS_BUDGET_EXCEEDED");
        record(loop.getInductionVar(), false);
        for (auto [arg, init] :
             llvm::zip(loop.getRegionIterArgs(), loop.getInitArgs()))
          record(arg, plan.isVarying(init));
        bool stable = false;
        for (unsigned round = 0; round <= loop.getNumRegionIterArgs();
             ++round) {
          if (!block(*loop.getBody(), depth + 1))
            return false;
          stable = true;
          auto yield = cast<scf::YieldOp>(loop.getBody()->back());
          for (auto [arg, value] :
               llvm::zip(loop.getRegionIterArgs(), yield.getOperands())) {
            if (plan.isVarying(value) && !plan.isVarying(arg)) {
              record(arg, true);
              stable = false;
            }
          }
          if (stable)
            break;
        }
        if (!stable)
          return reject(&op, "LOOP_VARIATION_DID_NOT_CONVERGE");
        for (auto [result, arg] :
             llvm::zip(loop.getResults(), loop.getRegionIterArgs()))
          record(result, plan.isVarying(arg));
        continue;
      }
      if (isa<triton::StoreOp>(op)) {
        for (Operation *parent = op.getParentOp(); parent;
             parent = parent->getParentOp())
          if (isa<scf::ForOp>(parent))
            return reject(&op, "LOOP_WRITE_NOT_IMPLEMENTED");
      }
      if (isa<triton::ReduceOp, triton::ScanOp>(op))
        for (Value capture : getBatchCombinerCaptures(&op)) {
          if (!known(capture, &op))
            return false;
          if (plan.isVarying(capture))
            return reject(&op, "UNSUPPORTED_REGION_CAPTURE");
        }
      bool varying = llvm::any_of(op.getOperands(),
                                  [&](Value v) { return plan.isVarying(v); });
      if (auto load = dyn_cast<triton::LoadOp>(op);
          load && load.getIsVolatile())
        varying = true;
      if (auto pid = dyn_cast<triton::GetProgramIdOp>(op))
        varying = static_cast<int>(pid.getAxis()) == axis;
      for (Value result : op.getResults())
        record(result, varying);
    }
    return true;
  }
};
} // namespace
BatchControlFlowPlan analyzeBatchControlFlow(triton::FuncOp function,
                                             int axis) {
  Analysis analysis(axis);
  for (Value arg : function.getArguments())
    analysis.record(arg, false);
  analysis.block(function.getBody().front());
  return std::move(analysis.plan);
}
} // namespace mlir::triton::cfg
