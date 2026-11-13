// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/ReductionModels.h"
#include "TritonToGraph/BlockBatching/BatchOpRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::triton::cfg {
namespace {
bool scalarNumerical(Type type) {
  return type.isF16() || type.isBF16() || type.isF32() || type.isInteger(1) ||
         type.isInteger(32) || type.isInteger(64);
}
} // namespace
bool hasBatchIndependentScanDomain(RankedTensorType source, int64_t axis,
                                   unsigned firstDimension) {
  if (!source || axis < int64_t(firstDimension) || axis >= source.getRank() ||
      source.getDimSize(axis) <= 0)
    return false;
  for (int64_t dim = firstDimension; dim < source.getRank(); ++dim)
    if (dim != axis && source.getDimSize(dim) != 1)
      return false;
  return true;
}

LogicalResult verifyBatchAggregationDomains(Operation *root,
                                            std::string &reason,
                                            Operation *&failedOp) {
  // A row reduction produces one independent value per surviving row. Track
  // those coordinates through existing shape/elementwise and SCF operations.
  // Collapsing one of them in another reduction needs a multi-axis tile
  // lowering, not the current independent-row lowering contract.
  struct Domain {
    llvm::SmallBitVector rows;
    bool unknown = false;
  };
  DenseMap<Value, Domain> domains;
  SmallVector<Value> pending;
  auto shape = [](Value value) -> ArrayRef<int64_t> {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    return type ? type.getShape() : ArrayRef<int64_t>{};
  };
  auto merge = [&](Value value, Domain incoming) {
    if (incoming.rows.none() && !incoming.unknown)
      return;
    auto &state = domains[value];
    state.rows.resize(shape(value).size());
    auto old = state.rows;
    bool changed = incoming.unknown && !state.unknown;
    state.unknown |= incoming.unknown;
    state.rows |= incoming.rows;
    if (changed || old != state.rows)
      pending.push_back(value);
  };
  root->walk([&](triton::ReduceOp reduce) {
    auto source = shape(reduce.getOperand(0));
    int64_t axis = reduce.getAxis();
    if (source[axis] == 1)
      return;
    for (Value output : reduce.getResults()) {
      Domain domain{llvm::SmallBitVector(shape(output).size()), false};
      for (auto [dim, extent] : llvm::enumerate(shape(output)))
        if (extent != 1)
          domain.rows.set(dim);
      merge(output, std::move(domain));
    }
  });
  unsigned budget = 65536;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    Domain input = domains.lookup(value);
    for (OpOperand &use : value.getUses()) {
      Operation *op = use.getOwner();
      auto reject = [&](StringRef why) {
        reason = why.str();
        failedOp = op;
        return failure();
      };
      if (!budget--)
        return reject("VV_AGGREGATION_ANALYSIS_BUDGET_EXCEEDED");
      if (auto loop = dyn_cast<scf::ForOp>(op)) {
        // Only iter_args carry tensors; bounds are scalar loop analysis facts.
        if (use.getOperandNumber() >= 3) {
          unsigned index = use.getOperandNumber() - 3;
          merge(loop.getRegionIterArgs()[index], input);
          merge(loop.getResult(index), input);
        }
        continue;
      }
      if (auto yield = dyn_cast<scf::YieldOp>(op)) {
        Operation *parent = yield->getParentOp();
        unsigned index = use.getOperandNumber();
        if (auto loop = dyn_cast<scf::ForOp>(parent)) {
          merge(loop.getRegionIterArgs()[index], input);
          merge(loop.getResult(index), input);
        } else if (auto branch = dyn_cast<scf::IfOp>(parent)) {
          merge(branch.getResult(index), input);
        }
        continue;
      }
      if (auto reduce = dyn_cast<triton::ReduceOp>(op)) {
        unsigned axis = reduce.getAxis();
        if (input.unknown)
          return reject("VV_AGGREGATION_DOMAIN_NOT_PROVEN");
        if (input.rows.test(axis) && shape(value)[axis] != 1)
          return reject("VV_MULTI_AXIS_REDUCTION_NOT_SUPPORTED");
        Domain output{llvm::SmallBitVector(input.rows.size() - 1), false};
        for (unsigned dim = 0, target = 0; dim < input.rows.size(); ++dim)
          if (dim != axis)
            output.rows[target++] = input.rows[dim];
        for (Value result : reduce.getResults())
          merge(result, output);
        continue;
      }
      // Memory observations and writes terminate compute-value provenance.
      if (isa<triton::LoadOp, triton::StoreOp>(op))
        continue;
      const auto *model = lookupBatchOp(op->getName().getStringRef());
      for (Value result : op->getResults()) {
        auto target = shape(result);
        Domain output{llvm::SmallBitVector(target.size()), input.unknown};
        if (auto expand = dyn_cast<triton::ExpandDimsOp>(op)) {
          for (unsigned dim = 0; dim < input.rows.size(); ++dim)
            output.rows[dim + (dim >= expand.getAxis())] = input.rows[dim];
        } else if (auto trans = dyn_cast<triton::TransOp>(op)) {
          for (unsigned dim = 0; dim < target.size(); ++dim)
            output.rows[dim] = input.rows[trans.getOrder()[dim]];
        } else if (isa<triton::ReshapeOp>(op)) {
          // Singleton-only reshapes preserve effective coordinates. General
          // flattening/splitting requires a separate coordinate proof.
          SmallVector<unsigned> from, to;
          for (auto [dim, extent] : llvm::enumerate(shape(value)))
            if (extent != 1)
              from.push_back(dim);
          for (auto [dim, extent] : llvm::enumerate(target))
            if (extent != 1)
              to.push_back(dim);
          bool same = from.size() == to.size();
          for (unsigned i = 0; same && i < from.size(); ++i)
            same = shape(value)[from[i]] == target[to[i]];
          if (same) {
            for (unsigned i = 0; i < from.size(); ++i)
              output.rows[to[i]] = input.rows[from[i]];
          } else {
            output.unknown = true;
          }
        } else if (target.size() == input.rows.size() &&
                   (isa<triton::BroadcastOp>(op) ||
                    (model && (model->model == BatchModel::Elementwise ||
                               model->model == BatchModel::IndexCast)))) {
          output.rows = input.rows;
        } else {
          output.unknown = true;
        }
        merge(result, std::move(output));
      }
    }
  }
  return success();
}

unsigned lowerBatchLaneScans(ModuleOp module, int64_t factor) {
  auto owner = module->getAttrOfType<StringAttr>("hacc.block_mapping_owner");
  if (!owner || owner.getValue() != "block_batching" || factor <= 1 ||
      factor > 64)
    return 0;
  auto *context = module.getContext();
  context->getOrLoadDialect<func::FuncDialect>();
  context->getOrLoadDialect<tensor::TensorDialect>();
  SmallVector<func::CallOp> calls;
  module.walk([&](func::CallOp call) { calls.push_back(call); });
  IRRewriter rewriter(context);
  SymbolTable symbols(module);
  llvm::SmallPtrSet<Operation *, 4> oldDeclarations;
  unsigned changed = 0;
  for (func::CallOp call : calls) {
    auto callee = symbols.lookup<func::FuncOp>(call.getCallee());
    auto parent = call->getParentOfType<func::FuncOp>();
    if (!callee || !callee.isExternal() || !callee.isPrivate() || !parent ||
        !callee.getSymName().starts_with("triton_cumsum") ||
        call.getNumOperands() != 3 || call.getNumResults() != 1)
      continue;
    auto mode = parent->getAttrOfType<StringAttr>("mix_mode");
    auto parallel = parent->getAttrOfType<StringAttr>("parallel_mode");
    if (!mode || mode.getValue() != "aiv" || !parallel ||
        (parallel.getValue() != "simd" &&
         parallel.getValue() != "mix_simd_simt"))
      continue;
    auto source = dyn_cast<RankedTensorType>(call.getOperand(0).getType());
    if (!source || source.getRank() < 2 || !source.hasStaticShape() ||
        source.getEncoding() || source.getDimSize(0) != factor ||
        source.getDimSize(1) <= 0 || call.getResult(0).getType() != source ||
        !(source.getElementType().isF32() ||
          source.getElementType().isInteger(32)) ||
        !call.getOperand(1).getType().isInteger(32) ||
        !call.getOperand(2).getType().isInteger(1))
      continue;
    APInt axis;
    if (!matchPattern(call.getOperand(1), m_ConstantInt(&axis)) ||
        axis.getSExtValue() <= 0 || axis.getSExtValue() >= source.getRank())
      continue;
    int64_t scanAxis = axis.getSExtValue();
    if (!hasBatchIndependentScanDomain(source, scanAxis, 1))
      continue;

    // Each row is an original program's scan. Keep its axis, order and
    // direction independent of the new batch dimension; do not flatten a
    // scan across lanes. This avoids the backend's 2D transpose-based scan.
    auto row = RankedTensorType::get({source.getDimSize(scanAxis)},
                                     source.getElementType());
    auto signature = rewriter.getFunctionType(
        {row, rewriter.getI32Type(), rewriter.getI1Type()}, {row});
    rewriter.setInsertionPoint(call);
    auto loc = call.getLoc();
    Value zeroAxis = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    SmallVector<OpFoldResult> sizes(source.getRank(), rewriter.getIndexAttr(1));
    sizes[scanAxis] = rewriter.getIndexAttr(source.getDimSize(scanAxis));
    SmallVector<OpFoldResult> strides(source.getRank(),
                                      rewriter.getIndexAttr(1));
    Value assembled = call.getOperand(0);
    for (int64_t lane = 0; lane < factor; ++lane) {
      func::FuncOp declaration;
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        declaration = rewriter.create<func::FuncOp>(
            loc, "triton_cumsum_bb_lane", signature);
        declaration.setPrivate();
        symbols.insert(declaration);
      }
      // The downstream adapter erases a cumulative-call declaration after
      // consuming one call. Give each lane its own symbol, including when
      // several scans in this module had shared the original declaration.
      SmallVector<OpFoldResult> offsets(source.getRank(),
                                        rewriter.getIndexAttr(0));
      offsets[0] = rewriter.getIndexAttr(lane);
      Value input = rewriter.create<tensor::ExtractSliceOp>(
          loc, row, call.getOperand(0), offsets, sizes, strides);
      Value scanned = rewriter
                          .create<func::CallOp>(
                              loc, declaration.getSymName(), TypeRange{row},
                              ValueRange{input, zeroAxis, call.getOperand(2)})
                          .getResult(0);
      assembled = rewriter.create<tensor::InsertSliceOp>(
          loc, scanned, assembled, offsets, sizes, strides);
    }
    rewriter.replaceOp(call, assembled);
    parent->setAttr("parallel_mode", rewriter.getStringAttr("mix_simd_simt"));
    // Cumsum's external template contains SIMT work that VF-mode inference
    // cannot see. Its D-cache reservation must participate in memory planning;
    // otherwise a SIMD-sized UB allocation can compile but fault at runtime.
    // Carry the downstream dialect attribute without coupling this analysis
    // library to HIVM. The backend parses it before inferring VF mode.
    parent->setAttr("hivm.vf_mode",
                    OpaqueAttr::get(rewriter.getStringAttr("hivm"),
                                    "vf_mode<MIX>", rewriter.getNoneType()));
    oldDeclarations.insert(callee.getOperation());
    ++changed;
  }
  for (Operation *declaration : oldDeclarations)
    if (SymbolTable::symbolKnownUseEmpty(declaration, module)) {
      symbols.remove(declaration);
      rewriter.eraseOp(declaration);
    }
  return changed;
}

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
