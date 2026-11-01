// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchAccessAnalysis.h"
#include "TritonToGraph/BlockBatching/BatchLayoutPlan.h"
#include "TritonToGraph/BlockBatching/BatchOpRegistry.h"
#include "TritonToGraph/BlockBatching/BatchPlan.h"
#include "TritonToGraph/BlockBatching/BatchValueMapping.h"
#include "TritonToGraph/BlockBatching/ControlFlowModels.h"
#include "TritonToGraph/BlockBatching/ElementwiseModels.h"
#include "TritonToGraph/BlockBatching/MemoryModels.h"
#include "TritonToGraph/BlockBatching/ReductionModels.h"
#include "TritonToGraph/BlockBatching/ShapeModels.h"
#include "TritonToGraph/ProgramGridTransform.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MathExtras.h"
#include <functional>

namespace mlir::triton::cfg {
namespace {
using J = llvm::json::Value;
using A = llvm::json::Array;

Type scalar(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return tensor.getElementType();
  return type;
}
bool supportedType(Type type, unsigned indexBits = 0) {
  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    if (tensor.getEncoding() || !tensor.hasStaticShape())
      return false;
    int64_t elements = 1;
    for (int64_t extent : tensor.getShape()) {
      if (extent <= 0 || extent > (1 << 20) / elements)
        return false;
      elements *= extent;
    }
    type = tensor.getElementType();
    if (type.isIndex())
      return false; // Tensor index needs a separate legal representation.
  }
  if (type.isIndex())
    return indexBits == 64;
  if (auto ptr = dyn_cast<triton::PointerType>(type)) {
    if (ptr.getAddressSpace() != 1)
      return false;
    type = ptr.getPointeeType();
  }
  if (auto integer = dyn_cast<IntegerType>(type))
    return integer.getWidth() == 1 || integer.getWidth() == 8 ||
           integer.getWidth() == 32 || integer.getWidth() == 64;
  return type.isF16() || type.isBF16() || type.isF32();
}

bool hasByteType(Type type) {
  type = scalar(type);
  if (auto ptr = dyn_cast<triton::PointerType>(type))
    type = ptr.getPointeeType();
  return type.isInteger(8);
}
bool byteOperationSupported(Operation *op) {
  bool usesByte = llvm::any_of(op->getOperandTypes(), hasByteType) ||
                  llvm::any_of(op->getResultTypes(), hasByteType);
  if (!usesByte)
    return true;
  auto name = op->getName().getStringRef();
  if (name == "tt.bitcast") {
    if (isBatchBoolToByteCast(op)) {
      // Only the frontend's explicit, byte-stored bool load contract is
      // admitted. Arbitrary ptr<i1> arithmetic is not assigned a byte size.
      return llvm::all_of(op->getUsers(), [](Operation *user) {
        return isBatchNormalizedBoolLoad(user);
      });
    }
    auto source =
        dyn_cast<triton::PointerType>(scalar(op->getOperand(0).getType()));
    auto result =
        dyn_cast<triton::PointerType>(scalar(op->getResult(0).getType()));
    return source && result && source.getPointeeType().isInteger(8) &&
           result.getPointeeType().isInteger(8);
  }
  return name == "arith.constant" || name == "arith.cmpi" ||
         name == "arith.select" || name == "tt.splat" || name == "tt.unsplat" ||
         name == "tt.expand_dims" || name == "tt.broadcast" ||
         name == "tt.reshape" || name == "tt.trans" || name == "tt.join" ||
         name == "tt.split" || name == "tt.addptr" || name == "tt.load" ||
         name == "tt.store";
}
bool attrsSupported(Operation *op) {
  const auto *descriptor = lookupBatchOp(op->getName().getStringRef());
  return descriptor && succeeded(verifyBatchAttributes(op, *descriptor));
}

// axis=None reductions flatten even rank-one inputs with allow_reorder.
// Choosing identity is legal; carrying that permission into the lifted type
// would instead permit permutations across independent program lanes.
void normalizeIdentityReductionReshapes(triton::FuncOp function) {
  SmallVector<triton::ReshapeOp> identities;
  function.walk([&](triton::ReshapeOp reshape) {
    auto type = dyn_cast<RankedTensorType>(reshape.getType());
    if (!type || type.getRank() != 1 || type.getEncoding() ||
        !type.hasStaticShape() || type != reshape.getSrc().getType() ||
        !reshape.getAllowReorder() || reshape->hasAttr("efficient_layout") ||
        !attrsSupported(reshape) || !reshape->hasOneUse())
      return;
    auto reduce = dyn_cast<triton::ReduceOp>(*reshape->getUsers().begin());
    if (reduce && reduce.getAxis() == 0)
      identities.push_back(reshape);
  });
  for (auto reshape : identities) {
    reshape.getResult().replaceAllUsesWith(reshape.getSrc());
    reshape.erase();
  }
}

// This proof authorizes no reordering or synchronization: the store's value
// already consumes the read through ordinary SSA dataflow. Do not infer a
// dependency from source order, a shared root, another load, or a region result.
bool hasPureDataDependency(Value value, Value read) {
  SmallVector<Value> pending{value};
  llvm::DenseSet<Value> visited;
  unsigned budget = 8192;
  while (!pending.empty() && budget--) {
    Value current = pending.pop_back_val();
    if (current == read)
      return true;
    if (!visited.insert(current).second)
      continue;
    Operation *producer = current.getDefiningOp();
    if (producer && producer->getNumRegions() == 0 &&
        isMemoryEffectFree(producer))
      llvm::append_range(pending, producer->getOperands());
  }
  return false;
}
} // namespace

static FailureOr<BatchRewriteReport> rewriteImpl(ModuleOp module,
                                                 BatchRequest request) {
  if (!request.isValid())
    return failure();
  BatchRewriteReport report;
  unsigned indexBits = getBatchIndexBitWidth(module);
  Operation *diagnosticOp = module.getOperation();
  auto reject = [&](llvm::StringRef reason) -> FailureOr<BatchRewriteReport> {
    setBatchDiagnostic(report, diagnosticOp, reason);
    return report;
  };
  if (request.factor == 1) {
    report.status = "DISABLED";
    report.reason = "DISABLED";
    return report;
  }
  if (request.factor > 64)
    return reject("FACTOR_EXCEEDS_LANE_CAPACITY");
  if (module->hasAttr("hacc.block_mapping_owner") ||
      module->hasAttr(kProgramGridTransformsAttr) ||
      module->hasAttr("hacc.coalesce_factor"))
    return reject("ALREADY_MAPPED");
  for (Operation &op : *module.getBody())
    if (!isa<triton::FuncOp>(op))
      return reject("UNSUPPORTED_MODULE_OPERATION");
  triton::FuncOp function;
  for (auto f : module.getOps<triton::FuncOp>()) {
    diagnosticOp = f.getOperation();
    if (function || !f.isPublic() || f.getBody().getBlocks().size() != 1)
      return reject("EXPECTED_SINGLE_PUBLIC_ENTRY");
    function = f;
  }
  if (!function || function.getFunctionType().getNumResults())
    return reject("UNSUPPORTED_ENTRY_SIGNATURE");
  for (Value arg : function.getArguments())
    if (isa<RankedTensorType>(arg.getType()) || arg.getType().isIndex() ||
        arg.getType().isInteger(8) || !supportedType(arg.getType(), indexBits))
      return reject("UNSUPPORTED_ENTRY_SIGNATURE");
  if (function->hasAttr("hacc.block_mapping_owner"))
    return reject("ALREADY_MAPPED");
  if (auto attr = function->getAttr("auto_blockify_size")) {
    auto size = dyn_cast<IntegerAttr>(attr);
    if (!size || size.getInt() != 1)
      return reject("G0_MAPPING_CONFLICT");
  }
  std::string normalizationReason;
  if (failed(normalizeBatchMemory(module, diagnosticOp, normalizationReason)))
    return reject(normalizationReason);
  normalizeIdentityReductionReshapes(function);
  auto &block = function.getBody().front();
  SmallVector<Operation *> originals;
  bool axes[3] = {};
  SmallVector<Operation *> validated;
  bool regionBudgetExceeded = false;
  std::function<void(Block &, unsigned)> collect = [&](Block &body,
                                                       unsigned depth) {
    if (depth > 8 || validated.size() > 8192) {
      regionBudgetExceeded = true;
      return;
    }
    for (Operation &op : body) {
      validated.push_back(&op);
      if (isa<scf::IfOp, scf::ForOp>(op))
        for (Region &region : op.getRegions())
          for (Block &nested : region)
            collect(nested, depth + 1);
    }
  };
  for (Operation &op : block)
    originals.push_back(&op);
  collect(block, 0);
  if (regionBudgetExceeded)
    return reject("REGION_DEPTH_EXCEEDED");
  for (Operation *operation : validated) {
    Operation &op = *operation;
    diagnosticOp = &op;
    auto name = op.getName().getStringRef();
    ++report.operations[name.str()];
    if (name.starts_with("tt.atomic") || name == "tt.print" ||
        name == "tt.assert")
      return reject("UNSUPPORTED_EFFECT");
    if (name == "tt.call")
      return reject("UNSUPPORTED_CALL");
    const auto *descriptor = lookupBatchOp(name);
    if (!descriptor)
      return reject("UNSUPPORTED_BATCH_OP");
    if (!descriptor->implemented)
      return reject(descriptor->specLevel == 'B' ? "BASIC_OP_NOT_IMPLEMENTED"
                                                 : "UNSUPPORTED_BATCH_OP");
    if (!attrsSupported(&op))
      return reject(name == "tt.extern_elementwise"
                        ? "UNSUPPORTED_EXTERNAL_FUNCTION"
                        : "UNSUPPORTED_OPERATION_ATTRIBUTE");
    if (!byteOperationSupported(&op))
      return reject("UNSUPPORTED_BYTE_OPERATION");
    if (name == "tt.reshape" && op.hasAttr("allow_reorder"))
      return reject("CROSS_LANE_REORDER");
    if (name == "tt.reshape" && op.hasAttr("efficient_layout"))
      return reject("UNSUPPORTED_LAYOUT_HINT");
    for (Type t : op.getOperandTypes())
      if (!supportedType(t, indexBits))
        return reject(t.isIndex() ? "INDEX_LAYOUT_NOT_PROVEN"
                                  : "UNSUPPORTED_TYPE_OR_RANK");
    for (Type t : op.getResultTypes()) {
      if (!supportedType(t, indexBits))
        return reject(t.isIndex() ? "INDEX_LAYOUT_NOT_PROVEN"
                                  : "UNSUPPORTED_TYPE_OR_RANK");
      if (auto tensor = dyn_cast<RankedTensorType>(t))
        if (tensor.getNumElements() >
            static_cast<int64_t>((1 << 20) / request.factor))
          return reject("TENSOR_SHAPE_BUDGET_EXCEEDED");
    }
    if (name == "tt.bitcast")
      if (isa<triton::PointerType>(scalar(op.getOperand(0).getType())) !=
          isa<triton::PointerType>(scalar(op.getResult(0).getType())))
        return reject("UNSUPPORTED_POINTER_BITCAST");
    if (auto pid = dyn_cast<triton::GetProgramIdOp>(op)) {
      int axis = static_cast<int>(pid.getAxis());
      if (axis < 0 || axis > 2)
        return reject("INVALID_PROGRAM_AXIS");
      axes[axis] = true;
    }
    if (auto count = dyn_cast<triton::GetNumProgramsOp>(op)) {
      int axis = static_cast<int>(count.getAxis());
      if (axis < 0 || axis > 2)
        return reject("INVALID_PROGRAM_AXIS");
    }
    if (isa<triton::ReduceOp, triton::ScanOp>(op)) {
      std::string reason;
      Operation *failedOp = nullptr;
      if (failed(verifyBatchReduction(&op, reason, failedOp))) {
        diagnosticOp = failedOp;
        return reject(reason);
      }
      op.walk([&](Operation *nested) {
        if (nested != &op)
          ++report.operations[nested->getName().getStringRef().str()];
      });
    } else if (op.getNumRegions() && !isa<scf::IfOp, scf::ForOp>(op))
      return reject("BASIC_CONTROL_FLOW_NOT_IMPLEMENTED");
  }
  // The runtime guard proves every original axis. Prefer the first used axis
  // for deterministic row batching; column masks spanning [F,D] need a more
  // general non-rectangular mask lowering and can still be requested
  // explicitly.
  int axis = request.axis;
  if (axis < 0)
    for (int i = 0; i < 3; ++i)
      if (axes[i]) {
        axis = i;
        break;
      }
  if (axis < 0)
    return reject("NO_PROGRAM_AXIS");
  if (!axes[axis])
    return reject("UNUSED_BATCH_AXIS");
  BatchPlan plan;
  auto &access = plan.effects;
  access = analyzeBatchAccesses(function, indexBits);
  if (!access.reason.empty()) {
    if (access.operation)
      diagnosticOp = access.operation;
    return reject(access.reason);
  }
  for (const auto &fact : access.accesses)
    if (fact.deviceAddress)
      plan.replicatedLoads.insert(fact.operation);
  auto &control = plan.values;
  control = analyzeBatchControlFlow(function, axis, plan.replicatedLoads);
  if (!control.reason.empty()) {
    diagnosticOp = control.operation;
    return reject(control.reason);
  }
  auto guard = llvm::json::parse(access.guardJSON);
  if (!guard || !guard->getAsObject())
    return module.emitError("invalid generated access facts"), failure();
  auto &object = *guard->getAsObject();
  object["version"] = 3;
  object["plan_kind"] = "unified";
  object["semantic_contract_version"] = 4;
  object["memory_policy"] = "disjoint_or_terminal_store";
  object["factor"] = request.factor;
  object["axis"] = axis;
  llvm::json::Array obligations;
  for (auto [id, fact] : llvm::enumerate(access.accesses)) {
    const char *rule = fact.deviceAddress ? "ReplicateEffect"
                       : fact.isVolatile ? "PreserveVolatile"
                       : !fact.write && !control.isVarying(fact.operation->getResult(0))
                           ? "ShareInvariantLoad"
                       : fact.write ? "SupportedEffect" : "ReplicateEffect";
    obligations.push_back(llvm::json::Object{
        {"access_id", static_cast<int64_t>(id)}, {"rule", rule},
        {"proof", (fact.write || fact.isVolatile || fact.normalizedPointer ||
                    !control.isVarying(fact.operation->getResult(0)))
                       ? "byte_domain" : "lane_correspondence"}});
  }
  // Known intervals retain the runtime disjointness proof. An opaque read
  // needs no host range when its only observable write is a terminal store
  // whose value consumes that read. Independent outputs, volatile effects,
  // region-local reads and writes preceding a read require a stronger proof.
  SmallVector<triton::StoreOp> writes;
  bool hasVolatile = false;
  for (const auto &fact : access.accesses) {
    if (fact.write)
      writes.push_back(cast<triton::StoreOp>(fact.operation));
    hasVolatile |= fact.isVolatile;
  }
  llvm::json::Array terminalReads;
  if (!writes.empty())
    for (auto [id, fact] : llvm::enumerate(access.accesses))
      if (fact.deviceAddress) {
        diagnosticOp = fact.operation;
        auto read = cast<triton::LoadOp>(fact.operation);
        if (writes.size() != 1 || hasVolatile ||
            read->getBlock() != &block || writes.front()->getBlock() != &block ||
            !read->isBeforeInBlock(writes.front().getOperation()) ||
            !hasPureDataDependency(writes.front().getValue(), read.getResult()))
          return reject("UNPROVEN_MEMORY_NONINTERFERENCE");
        terminalReads.push_back(static_cast<int64_t>(id));
      }
  object["terminal_read_ids"] = std::move(terminalReads);
  object["obligations"] = std::move(obligations);
  report.guardJSON = llvm::formatv("{0}", *guard).str();
  auto &layouts = plan.layouts;
  layouts = planBatchLayouts(access, axis, request.factor);
  report.layoutJSON = serializeBatchLayouts(layouts);
  llvm::DenseMap<Operation *, unsigned> accessIDs;
  SmallVector<Operation *> semanticAccesses(access.accesses.size(), nullptr);
  for (auto [id, fact] : llvm::enumerate(access.accesses))
    accessIDs[fact.operation] = id;
  report.selectedAxis = axis;
  report.laneWidth = request.factor;
  const int64_t lanes = report.laneWidth;
  const unsigned originalArgCount = function.getNumArguments();
  if (failed(addProgramGridHiddenExtentArguments(function, 3)))
    return reject("HIDDEN_ARGUMENT_CONFLICT");

  IRRewriter rewriter(module.getContext());
  rewriter.setInsertionPointToStart(&block);
  Location loc = function.getLoc();
  auto i32 = rewriter.getI32Type();
  auto i64 = rewriter.getI64Type();
  auto lane32 = RankedTensorType::get({lanes}, i32);
  auto lane64 = RankedTensorType::get({lanes}, i64);
  auto constant = [&](Type t, int64_t n) -> Value {
    TypedAttr attr = rewriter.getIntegerAttr(scalar(t), n);
    if (auto tensor = dyn_cast<RankedTensorType>(t))
      attr = DenseElementsAttr::get(tensor, attr);
    return rewriter.create<arith::ConstantOp>(loc, t, attr);
  };
  Operation *pidSource = nullptr;
  for (auto *op : validated)
    if (auto pid = dyn_cast<triton::GetProgramIdOp>(op);
        pid && static_cast<int>(pid.getAxis()) == axis) {
      pidSource = op;
      break;
    }
  Value group = rewriter.clone(*pidSource)->getResult(0);
  Value group64 = rewriter.create<arith::ExtSIOp>(loc, i64, group);
  Value lane = rewriter.create<triton::MakeRangeOp>(loc, lane32, 0, lanes);
  Value laneExt = rewriter.create<arith::ExtSIOp>(loc, lane64, lane);
  Value base64 = rewriter.create<arith::MulIOp>(loc, group64,
                                                constant(i64, request.factor));
  Value baseVec = rewriter.create<triton::SplatOp>(loc, lane64, base64);
  Value p64 = rewriter.create<arith::AddIOp>(loc, baseVec, laneExt);
  Value g64 = rewriter.create<arith::ExtSIOp>(
      loc, i64, function.getArgument(originalArgCount + axis));
  Value gVec = rewriter.create<triton::SplatOp>(loc, lane64, g64);
  // Lane storage is exactly F, including non-power-of-two factors. Only the
  // final original-grid group has newly inactive lanes.
  Value live =
      rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, p64, gVec);
  // A nonempty launch guarantees group*F <= G-1 <= INT32_MAX.
  // Keep the lane range affine so normal strided memory lowering can recover
  // contiguous columns. Inactive lanes may wrap in ordinary integer ops (no
  // poison flags are admitted), but every varying memory effect is live-masked.
  Value base32 = rewriter.create<arith::TruncIOp>(loc, i32, base64);
  Value base32Vec = rewriter.create<triton::SplatOp>(loc, lane32, base32);
  Value pid = rewriter.create<arith::AddIOp>(loc, base32Vec, lane);
  BatchValueMapping values;
  IRMapping &mapping = values.mapping;
  for (Value arg : function.getArguments())
    values.map(arg, arg, false);
  auto liftedType = [&](Type t) -> RankedTensorType {
    SmallVector<int64_t> shape{lanes};
    if (auto tensor = dyn_cast<RankedTensorType>(t))
      llvm::append_range(shape, tensor.getShape());
    return RankedTensorType::get(shape, scalar(t));
  };
  // Reuse only within the current insertion block. A value materialized in
  // one if arm must never be reused in its sibling or after the join.
  llvm::DenseMap<Block *, llvm::DenseMap<std::pair<Value, Type>, Value>>
      liftedValues;
  llvm::DenseMap<Block *, llvm::DenseMap<Type, Value>> liveMasks;
  auto lift = [&](Value old, RankedTensorType target) -> Value {
    Value value = mapping.lookup(old);
    if (value.getType() == target)
      return value;
    auto &cache = liftedValues[rewriter.getInsertionBlock()];
    auto key = std::make_pair(old, Type(target));
    if (auto found = cache.find(key); found != cache.end())
      return found->second;
    auto tensor = dyn_cast<RankedTensorType>(value.getType());
    if (!tensor) {
      Value result = rewriter.create<triton::SplatOp>(loc, target, value);
      cache[key] = result;
      return result;
    }
    while (cast<RankedTensorType>(value.getType()).getRank() <
           target.getRank()) {
      int axis = values.isVarying(old)
                     ? cast<RankedTensorType>(value.getType()).getRank()
                     : 0;
      value = rewriter.create<triton::ExpandDimsOp>(loc, value, axis);
    }
    if (value.getType() != target)
      value = rewriter.create<triton::BroadcastOp>(loc, target, value);
    cache[key] = value;
    return value;
  };
  auto liveFor = [&](RankedTensorType target) -> Value {
    auto maskType =
        RankedTensorType::get(target.getShape(), rewriter.getI1Type());
    auto &cache = liveMasks[rewriter.getInsertionBlock()];
    if (auto found = cache.find(maskType); found != cache.end())
      return found->second;
    Value mask = live;
    while (cast<RankedTensorType>(mask.getType()).getRank() < target.getRank())
      mask = rewriter.create<triton::ExpandDimsOp>(
          loc, mask, cast<RankedTensorType>(mask.getType()).getRank());
    if (mask.getType() != maskType)
      mask = rewriter.create<triton::BroadcastOp>(loc, maskType, mask);
    cache[maskType] = mask;
    return mask;
  };
  auto rejectRewrite = [&](llvm::StringRef reason) -> LogicalResult {
    setBatchDiagnostic(report, diagnosticOp, reason);
    return failure();
  };
  std::function<LogicalResult(Block &, Block *, TypeRange)> rewriteBody;
  rewriteBody = [&](Block &source, Block *target,
                    TypeRange yieldTypes) -> LogicalResult {
    SmallVector<Operation *> sourceOps;
    if (!target)
      sourceOps = originals;
    else
      for (Operation &original : source)
        sourceOps.push_back(&original);
    for (Operation *op : sourceOps) {
      if (target)
        rewriter.setInsertionPointToEnd(target);
      else
        rewriter.setInsertionPoint(op);
      diagnosticOp = op;
      loc = op->getLoc();
      if (auto oldPid = dyn_cast<triton::GetProgramIdOp>(op);
          oldPid && static_cast<int>(oldPid.getAxis()) == axis) {
        values.map(op->getResult(0), pid, true);
        continue;
      }
      if (auto programs = dyn_cast<triton::GetNumProgramsOp>(op)) {
        values.map(
            op->getResult(0),
            function.getArgument(originalArgCount +
                                 static_cast<unsigned>(programs.getAxis())),
            false);
        continue;
      }
      for (Value operand : op->getOperands())
        if (!values.contains(operand))
          return rejectRewrite("VALUE_MAPPING_MISSING");
      if (isa<scf::YieldOp>(op)) {
        if (op->getNumOperands() != yieldTypes.size())
          return rejectRewrite("REGION_YIELD_TYPE_MISMATCH");
        SmallVector<Value> yielded;
        for (auto [old, type] : llvm::zip(op->getOperands(), yieldTypes)) {
          Value value = mapping.lookup(old);
          if (value.getType() != type && isa<RankedTensorType>(type))
            value = lift(old, cast<RankedTensorType>(type));
          if (value.getType() != type)
            return rejectRewrite("REGION_YIELD_TYPE_MISMATCH");
          yielded.push_back(value);
        }
        rewriter.create<scf::YieldOp>(loc, yielded);
        continue;
      }
      if (isa<scf::IfOp, scf::ForOp>(op)) {
        SmallVector<Type> resultTypes;
        for (Value result : op->getResults())
          resultTypes.push_back(control.isVarying(result)
                                    ? Type(liftedType(result.getType()))
                                    : result.getType());
        Operation *replacement = nullptr;
        if (auto branch = dyn_cast<scf::IfOp>(op)) {
          auto created = rewriter.create<scf::IfOp>(
              loc, resultTypes, mapping.lookup(branch.getCondition()),
              !branch.getElseRegion().empty());
          created->setAttrs(op->getAttrs());
          for (auto [from, to] :
               llvm::zip(op->getRegions(), created->getRegions())) {
            if (from.empty())
              continue;
            Block &destination = to.front();
            while (!destination.empty())
              rewriter.eraseOp(&destination.back());
            if (failed(rewriteBody(from.front(), &destination, resultTypes)))
              return failure();
          }
          replacement = created;
        } else {
          auto loop = cast<scf::ForOp>(op);
          SmallVector<Value> init;
          for (auto [value, type] : llvm::zip(loop.getInitArgs(), resultTypes))
            init.push_back(value.getType() == type && !values.isVarying(value)
                               ? mapping.lookup(value)
                           : isa<RankedTensorType>(type)
                               ? lift(value, cast<RankedTensorType>(type))
                               : mapping.lookup(value));
          // A backend loop-invariant-yield fold can replace a tensor result
          // even on a zero-trip loop. Express the original zero-trip result
          // explicitly, while leaving the positive iteration sequence intact.
          Value nonempty = rewriter.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::slt,
              mapping.lookup(loop.getLowerBound()),
              mapping.lookup(loop.getUpperBound()));
          auto gate =
              rewriter.create<scf::IfOp>(loc, resultTypes, nonempty, true);
          Block &thenBlock = gate.getThenRegion().front();
          Block &elseBlock = gate.getElseRegion().front();
          while (!thenBlock.empty())
            rewriter.eraseOp(&thenBlock.back());
          while (!elseBlock.empty())
            rewriter.eraseOp(&elseBlock.back());
          rewriter.setInsertionPointToEnd(&thenBlock);
          auto created = rewriter.create<scf::ForOp>(
              loc, mapping.lookup(loop.getLowerBound()),
              mapping.lookup(loop.getUpperBound()),
              mapping.lookup(loop.getStep()), init);
          created->setAttrs(op->getAttrs());
          while (!created.getBody()->empty())
            rewriter.eraseOp(&created.getBody()->back());
          values.map(loop.getInductionVar(), created.getInductionVar(), false);
          for (auto [from, to] :
               llvm::zip(loop.getRegionIterArgs(), created.getRegionIterArgs()))
            values.map(from, to, control.isVarying(from));
          if (failed(
                  rewriteBody(*loop.getBody(), created.getBody(), resultTypes)))
            return failure();
          rewriter.setInsertionPointToEnd(&thenBlock);
          rewriter.create<scf::YieldOp>(op->getLoc(), created.getResults());
          rewriter.setInsertionPointToEnd(&elseBlock);
          rewriter.create<scf::YieldOp>(op->getLoc(), init);
          replacement = gate;
        }
        for (auto [old, result] :
             llvm::zip(op->getResults(), replacement->getResults()))
          values.map(old, result, control.isVarying(old));
        continue;
      }
      bool var = op->getNumResults()
                     ? control.isVarying(op->getResult(0))
                     : llvm::any_of(op->getOperands(), [&](Value value) {
                         return control.isVarying(value);
                       });
      if (isa<triton::ReduceOp, triton::ScanOp>(op))
        for (Value capture : getBatchCombinerCaptures(op)) {
          if (!values.contains(capture))
            return rejectRewrite("VALUE_MAPPING_MISSING");
          if (values.isVarying(capture))
            return rejectRewrite("UNSUPPORTED_REGION_CAPTURE");
        }
      if (var &&
          (llvm::any_of(op->getOperandTypes(),
                        [](Type type) { return scalar(type).isIndex(); }) ||
           llvm::any_of(op->getResultTypes(),
                        [](Type type) { return scalar(type).isIndex(); })))
        return rejectRewrite("VARYING_INDEX_NOT_IMPLEMENTED");
      // Every store must be lane-exclusive. The original-domain host proof also
      // checks axes not batched by this request.
      if (isa<triton::StoreOp>(op) && !var)
        return rejectRewrite("UNIFORM_STORE");
      Operation *replacement = nullptr;
      if (auto intToPtr = dyn_cast<triton::IntToPtrOp>(op)) {
        auto pointer =
            dyn_cast<triton::PointerType>(scalar(intToPtr.getType()));
        unsigned bytes =
            pointer ? pointer.getPointeeType().getIntOrFloatBitWidth() / 8 : 0;
        auto plan = getBatchPointerPlan(intToPtr.getSrc(), bytes);
        if (!plan || scalar(plan->base.getType()) != pointer)
          return rejectRewrite("POINTER_ROUNDTRIP_NOT_LOWERABLE");
        Type resultType =
            var ? Type(liftedType(intToPtr.getType())) : intToPtr.getType();
        auto tensor = dyn_cast<RankedTensorType>(resultType);
        auto integerType =
            tensor ? Type(RankedTensorType::get(tensor.getShape(), i64))
                   : Type(i64);
        auto mapped = [&](Value value, Type type) -> Value {
          if (!values.contains(value))
            return {};
          return isa<RankedTensorType>(type)
                     ? lift(value, cast<RankedTensorType>(type))
                     : mapping.lookup(value);
        };
        Value base = mapped(plan->base, resultType);
        if (!base)
          return rejectRewrite("VALUE_MAPPING_MISSING");
        Value offset = constant(integerType, 0);
        for (auto [delta, subtract] : plan->byteOffsets) {
          Value value = mapped(delta, integerType);
          if (!value || value.getType() != integerType)
            return rejectRewrite("POINTER_ROUNDTRIP_NOT_LOWERABLE");
          // Divisibility is proven on the original expression. The runtime
          // guard proves its signed range and the complete byte-address
          // provenance.
          Value elements = rewriter.create<arith::DivSIOp>(
              loc, value, constant(integerType, bytes));
          if (subtract)
            offset = rewriter.create<arith::SubIOp>(loc, offset, elements);
          else
            offset = rewriter.create<arith::AddIOp>(loc, offset, elements);
        }
        replacement =
            rewriter.create<triton::AddPtrOp>(loc, resultType, base, offset);
      } else if (!var || isa<triton::ReturnOp>(op)) {
        replacement = rewriter.clone(*op, mapping);
      } else if (auto splat = dyn_cast<triton::SplatOp>(op)) {
        Value result = lift(splat.getSrc(), liftedType(splat.getType()));
        values.map(splat.getResult(), result, true);
        continue;
      } else if (lookupBatchOp(op->getName().getStringRef())->model ==
                 BatchModel::Shape) {
        IRMapping operands = mapping;
        if (isa<triton::JoinOp>(op))
          for (Value operand : op->getOperands())
            operands.map(operand, lift(operand, liftedType(operand.getType())));
        SmallVector<Type> results;
        for (Type type : op->getResultTypes())
          results.push_back(liftedType(type));
        replacement = rewriteBatchShape(op, results, operands, rewriter);
      } else if (isa<triton::ReduceOp, triton::ScanOp>(op)) {
        SmallVector<Value> inputs;
        for (Value source : op->getOperands())
          inputs.push_back(lift(source, liftedType(source.getType())));
        SmallVector<Type> results;
        for (Type type : op->getResultTypes())
          results.push_back(liftedType(type));
        replacement =
            rewriteBatchReduction(op, inputs, results, mapping, rewriter);
      } else if (auto load = dyn_cast<triton::LoadOp>(op)) {
        auto type = liftedType(load.getType());
        Value ptr = lift(load.getPtr(),
                         RankedTensorType::get(
                             type.getShape(), scalar(load.getPtr().getType())));
        Value mask = liveFor(type);
        if (load.getMask())
          mask = rewriter.create<arith::AndIOp>(
              loc, mask,
              lift(load.getMask(), RankedTensorType::get(
                                       type.getShape(), rewriter.getI1Type())));
        Value other =
            load.getOther()
                ? lift(load.getOther(), type)
                : rewriter.create<arith::ConstantOp>(
                      loc, type,
                      DenseElementsAttr::get(
                          type, rewriter.getZeroAttr(type.getElementType())));
        replacement = rewriter.create<triton::LoadOp>(
            loc, ptr, mask, other, load.getBoundaryCheck(), load.getPadding(),
            load.getCache(), load.getEvict(), load.getIsVolatile());
        if (auto marker = load->getAttr("was_bool_to_int8"))
          replacement->setAttr("was_bool_to_int8", marker);
      } else if (auto store = dyn_cast<triton::StoreOp>(op)) {
        auto type = liftedType(store.getValue().getType());
        Value ptr =
            lift(store.getPtr(),
                 RankedTensorType::get(type.getShape(),
                                       scalar(store.getPtr().getType())));
        Value mask = liveFor(type);
        if (store.getMask())
          mask = rewriter.create<arith::AndIOp>(
              loc, mask,
              lift(store.getMask(),
                   RankedTensorType::get(type.getShape(),
                                         rewriter.getI1Type())));
        replacement = rewriter.create<triton::StoreOp>(
            loc, ptr, lift(store.getValue(), type), mask,
            store.getBoundaryCheck(), store.getCache(), store.getEvict());
      } else {
        if (op->getNumResults() != 1)
          return rejectRewrite("UNSUPPORTED_RESULT_SCHEMA");
        const auto *descriptor = lookupBatchOp(op->getName().getStringRef());
        if (!descriptor || (descriptor->model != BatchModel::Elementwise &&
                            op->getName().getStringRef() != "tt.addptr" &&
                            op->getName().getStringRef() != "tt.ptr_to_int" &&
                            op->getName().getStringRef() != "tt.int_to_ptr"))
          return rejectRewrite("BASIC_OP_NOT_IMPLEMENTED");
        SmallVector<int64_t> shape =
            llvm::to_vector(liftedType(op->getResult(0).getType()).getShape());
        SmallVector<Value> lifted;
        for (Value operand : op->getOperands())
          lifted.push_back(
              lift(operand,
                   RankedTensorType::get(shape, scalar(operand.getType()))));
        Value mask;
        if (needsBatchActiveMask(*descriptor))
          mask = liveFor(liftedType(op->getResult(0).getType()));
        auto safe = prepareBatchElementwiseOperands(*descriptor, lifted, mask,
                                                    loc, rewriter);
        replacement = rewriteBatchElementwise(
            op, *descriptor, safe, liftedType(op->getResult(0).getType()), loc,
            rewriter);
      }
      if (auto found = accessIDs.find(op); found != accessIDs.end())
        semanticAccesses[found->second] = replacement;
      for (auto [old, result] :
           llvm::zip(op->getResults(), replacement->getResults())) {
        Type expected = var ? Type(liftedType(old.getType())) : old.getType();
        if (result.getType() != expected)
          return rejectRewrite("VALUE_MAPPING_TYPE_MISMATCH");
        values.map(old, result, var);
      }
    }
    return success();
  };
  if (failed(rewriteBody(block, nullptr, {})))
    return report;
  // Verify effect correspondence while the original operations still exist.
  // Sharing changes dynamic multiplicity only under its emitted proof; every
  // source effect still owns exactly one generated operation at this stage.
  for (auto [id, fact] : llvm::enumerate(access.accesses)) {
    Operation *generated = semanticAccesses[id];
    if (!generated || generated->getName() != fact.operation->getName())
      return module.emitError("missing or invalid planned memory effect"), failure();
    if (auto load = dyn_cast<triton::LoadOp>(generated)) {
      auto source = cast<triton::LoadOp>(fact.operation);
      if (load.getCache() != source.getCache() ||
          load.getEvict() != source.getEvict() ||
          load.getIsVolatile() != source.getIsVolatile())
        return module.emitError("planned load policy changed"), failure();
    }
  }
  // All layout facts are owned expressions. Drop original operation handles
  // before erasing source IR; only the generated semantic side table survives.
  accessIDs.clear();
  control.varying.clear();
  plan.replicatedLoads.clear();
  for (auto &fact : access.accesses)
    fact.operation = nullptr;
  for (auto *op : llvm::reverse(originals))
    rewriter.eraseOp(op);
  ProgramGridTransformContract contract;
  contract.version = 3;
  contract.transforms.push_back({0, axis, request.factor, false, false});
  if (failed(setProgramGridTransformContract(module, contract)))
    return module.emitError(
               "block batching generated an invalid grid contract"),
           failure();
  module->setAttr("hacc.block_mapping_owner",
                  rewriter.getStringAttr("block_batching"));
  function->setAttr("hacc.block_mapping_owner",
                    rewriter.getStringAttr("block_batching"));
  if (failed(verify(module)))
    return module.emitError("block batching generated invalid TTIR"), failure();
  bool closureValid = true;
  module.walk([&](Operation *op) {
    if (!closureValid)
      return;
    auto name = op->getName().getStringRef();
    const auto *descriptor = lookupBatchOp(name);
    if (!descriptor || !descriptor->implemented ||
        (descriptor->model != BatchModel::Structural && !attrsSupported(op))) {
      setBatchDiagnostic(report, op, "GENERATED_OP_CLOSURE_INVALID");
      closureValid = false;
    }
  });
  if (!closureValid)
    return report;
  if (!request.semanticOnly &&
      failed(rewriteBatchMemoryViews(module, access, layouts, semanticAccesses,
                                     group, originalArgCount, axis,
                                     request.factor)))
    return failure();
  report.layoutJSON = serializeBatchLayouts(layouts);
  report.status = "REWRITTEN";
  llvm::json::Array layoutChecks;
  for (const auto &plan : layouts)
    if (plan.rewritten && !plan.runtimeGuardJSON.empty()) {
      auto parsed = llvm::json::parse(plan.runtimeGuardJSON);
      if (!parsed) {
        llvm::consumeError(parsed.takeError());
        return module.emitError("invalid generated layout guard"), failure();
      }
      layoutChecks.push_back(std::move(*parsed));
    }
  {
    auto parsed = llvm::json::parse(report.guardJSON);
    if (!parsed) {
      llvm::consumeError(parsed.takeError());
      return module.emitError("invalid original access guard"), failure();
    }
    if (!parsed->getAsObject())
      return module.emitError("invalid original access guard"), failure();
    auto *proofs = parsed->getAsObject()->getArray("obligations");
    if (!proofs || proofs->size() != layouts.size())
      return module.emitError("invalid effect proof mapping"), failure();
    for (const auto &layout : layouts)
      if (layout.rewritten)
        (*proofs)[layout.accessID].getAsObject()->operator[]("proof") = "byte_domain";
    if (!layoutChecks.empty())
      (*parsed->getAsObject())["layout_checks"] = std::move(layoutChecks);
    report.guardJSON = llvm::formatv("{0}", *parsed).str();
  }
  report.reason = "";
  return report;
}
FailureOr<BatchRewriteReport> rewriteBlockBatching(ModuleOp module,
                                                   BatchRequest request) {
  OwningOpRef<ModuleOp> sandbox(module.clone());
  auto result = rewriteImpl(sandbox.get(), request);
  if (succeeded(result) && result->status == "REWRITTEN") {
    ModuleOp candidate = sandbox.get();
    module->setAttrs(candidate->getAttrs());
    module.getBodyRegion().takeBody(candidate.getBodyRegion());
  }
  return result;
}
} // namespace mlir::triton::cfg
