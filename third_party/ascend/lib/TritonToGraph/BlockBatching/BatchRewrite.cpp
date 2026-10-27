// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchPlan.h"
#include "TritonToGraph/ProgramGridTransform.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::triton::cfg {
namespace {
using J = llvm::json::Value;
using A = llvm::json::Array;

// An explicit v0 closure. A pure trait alone is never an admission rule.
bool supported(llvm::StringRef n) {
  static const llvm::StringSet<> names{
      "arith.constant", "arith.addi", "arith.subi", "arith.muli",
      "arith.cmpi", "arith.andi", "arith.ori", "arith.select",
      "arith.extsi", "arith.extui", "arith.trunci", "arith.sitofp",
      "arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.negf",
      "arith.extf", "arith.truncf", "math.exp", "math.exp2", "math.sqrt",
      "math.rsqrt", "math.fma", "tt.fp_to_fp", "tt.precise_sqrt",
      "tt.precise_divf", "tt.get_program_id", "tt.get_num_programs",
      "tt.make_range", "tt.splat", "tt.addptr", "tt.load", "tt.store",
      "tt.reduce", "tt.return"};
  return names.contains(n);
}
Type scalar(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return tensor.getElementType();
  return type;
}
bool supportedType(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    if (tensor.getRank() != 1 || tensor.getEncoding() || !tensor.hasStaticShape() || tensor.getNumElements() <= 0)
      return false;
    type = tensor.getElementType();
  }
  if (auto ptr = dyn_cast<triton::PointerType>(type)) {
    if (ptr.getAddressSpace() != 1) return false;
    type = ptr.getPointeeType();
  }
  if (auto integer = dyn_cast<IntegerType>(type))
    return integer.getWidth() == 1 || integer.getWidth() == 32 || integer.getWidth() == 64;
  return type.isF16() || type.isBF16() || type.isF32();
}

// Serialize the *original* integer expressions, including their bit widths.
// The host proves their ranges and memory domains for every invocation.
struct AccessAnalysis {
  triton::FuncOp function;
  DenseMap<Value, J> expressions;
  llvm::json::Array accesses;
  DenseMap<Value, unsigned> expressionSizes;
  unsigned budget = 256;
  std::string reason;
  explicit AccessAnalysis(triton::FuncOp f) : function(f) {}

  J integer(Value value) {
    if (!budget) { reason = "ANALYSIS_BUDGET_EXCEEDED"; return nullptr; }
    --budget;
    if (auto it = expressions.find(value); it != expressions.end())
      return it->second;
    auto ty = dyn_cast<IntegerType>(scalar(value.getType()));
    if (!ty) { reason = "NON_INTEGER_ADDRESS"; return nullptr; }
    J result = nullptr;
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      if (arg.getOwner() != &function.getBody().front()) {
        reason = "UNSUPPORTED_INDEX_CAPTURE"; return nullptr;
      }
      result = A{"arg", arg.getArgNumber(), ty.getWidth()};
    } else if (auto c = value.getDefiningOp<arith::ConstantOp>()) {
      if (auto attr = dyn_cast<IntegerAttr>(c.getValue()))
        result = A{"const", attr.getInt(), ty.getWidth()};
      else if (auto attr = dyn_cast<DenseIntElementsAttr>(c.getValue()); attr && attr.isSplat())
        result = A{"const", attr.getSplatValue<APInt>().getSExtValue(), ty.getWidth()};
    } else if (auto pid = value.getDefiningOp<triton::GetProgramIdOp>()) {
      result = A{"pid", static_cast<int64_t>(pid.getAxis()), 32};
    } else if (auto grid = value.getDefiningOp<triton::GetNumProgramsOp>()) {
      result = A{"grid", static_cast<int64_t>(grid.getAxis()), 32};
    } else if (auto range = value.getDefiningOp<triton::MakeRangeOp>()) {
      result = A{"range", range.getStart(), range.getEnd(), 32};
    } else if (auto splat = value.getDefiningOp<triton::SplatOp>()) {
      result = integer(splat.getSrc());
    } else {
      Operation *op = value.getDefiningOp();
      auto name = op->getName().getStringRef();
      if (name == "arith.addi" || name == "arith.subi" || name == "arith.muli")
        result = A{name.drop_front(6).str(), ty.getWidth(), integer(op->getOperand(0)), integer(op->getOperand(1))};
      else if (name == "arith.extsi" || name == "arith.extui" || name == "arith.trunci")
        result = A{name.drop_front(6).str(), ty.getWidth(), integer(op->getOperand(0))};
    }
    if (result.kind() == J::Null && reason.empty())
      reason = "UNSUPPORTED_ADDRESS_EXPRESSION";
    unsigned expressionSize = 1;
    if (Operation *op = value.getDefiningOp())
      for (Value operand : op->getOperands())
        expressionSize += expressionSizes.lookup(operand);
    if (expressionSize > 256) {
      reason = "ANALYSIS_BUDGET_EXCEEDED"; result = nullptr;
    }
    expressionSizes[value] = expressionSize;
    expressions.try_emplace(value, result);
    return result;
  }

  std::pair<int, J> pointer(Value value) {
    if (!budget) { reason = "ANALYSIS_BUDGET_EXCEEDED"; return {-1, nullptr}; }
    --budget;
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      if (arg.getOwner() == &function.getBody().front() && isa<triton::PointerType>(arg.getType()))
        return {static_cast<int>(arg.getArgNumber()), A{"const", 0, 64}};
    }
    if (auto splat = value.getDefiningOp<triton::SplatOp>())
      return pointer(splat.getSrc());
    if (auto add = value.getDefiningOp<triton::AddPtrOp>()) {
      auto base = pointer(add.getPtr());
      if (base.first < 0) return base;
      return {base.first, A{"addi", 64, std::move(base.second), integer(add.getOffset())}};
    }
    reason = "UNKNOWN_POINTER_ROOT";
    return {-1, nullptr};
  }

  void add(Value ptr, Value mask, bool write) {
    auto root = pointer(ptr);
    auto pointerTy = dyn_cast<triton::PointerType>(scalar(ptr.getType()));
    if (root.first < 0 || !pointerTy || !pointerTy.getPointeeType().isIntOrFloat()) {
      reason = "UNSUPPORTED_POINTER_TYPE"; return;
    }
    J domain = nullptr;
    if (mask) {
      auto cmp = mask.getDefiningOp<arith::CmpIOp>();
      if (!cmp || (cmp.getPredicate() != arith::CmpIPredicate::slt &&
                   cmp.getPredicate() != arith::CmpIPredicate::ult)) {
        reason = "UNSUPPORTED_ACCESS_MASK"; return;
      }
      domain = A{"lt", integer(cmp.getLhs()), integer(cmp.getRhs())};
    }
    accesses.push_back(llvm::json::Object{
        {"root", root.first}, {"offset", std::move(root.second)},
        {"mask", std::move(domain)}, {"write", write},
        {"bytes", pointerTy.getPointeeType().getIntOrFloatBitWidth() / 8}});
  }
};

bool attrsSupported(Operation *op) {
  // Preserve only modeled attributes. Shape facts on lifted results must not
  // accidentally retain a one-dimensional contiguity/divisibility promise.
  static const llvm::StringSet<> allowed{"value", "axis", "start", "end", "predicate",
      "cache", "evict", "isVolatile", "operandSegmentSizes", "boundaryCheck",
      "padding", "rounding", "fp_downcast_rounding", "fastmath", "overflowFlags"};
  for (NamedAttribute attr : op->getAttrs()) {
    if (!allowed.contains(attr.getName().strref()))
      return false;
    if (attr.getName() == "fastmath") {
      auto flags = dyn_cast<arith::FastMathFlagsAttr>(attr.getValue());
      if (!flags || flags.getValue() != arith::FastMathFlags::none)
        return false;
    }
    if (attr.getName() == "overflowFlags") {
      auto flags = dyn_cast<arith::IntegerOverflowFlagsAttr>(attr.getValue());
      if (!flags || flags.getValue() != arith::IntegerOverflowFlags::none)
        return false;
    }
  }
  return true;
}
} // namespace

static FailureOr<BatchRewriteReport> rewriteImpl(ModuleOp module, BatchRequest request) {
  if (!request.isValid())
    return failure();
  BatchRewriteReport report;
  auto reject = [&](llvm::StringRef reason) -> FailureOr<BatchRewriteReport> {
    report.reason = reason.str(); return report;
  };
  if (request.factor == 1) {
    report.status = "DISABLED"; report.reason = "DISABLED"; return report;
  }
  if (request.factor > 64)
    return reject("FACTOR_EXCEEDS_LANE_CAPACITY");
  if (module->hasAttr("hacc.block_mapping_owner") || module->hasAttr(kProgramGridTransformsAttr) ||
      module->hasAttr("hacc.coalesce_factor"))
    return reject("ALREADY_MAPPED");
  for (Operation &op : *module.getBody())
    if (!isa<triton::FuncOp>(op)) return reject("UNSUPPORTED_MODULE_OPERATION");
  triton::FuncOp function;
  for (auto f : module.getOps<triton::FuncOp>()) {
    if (function || !f.isPublic() || f.getBody().getBlocks().size() != 1)
      return reject("EXPECTED_SINGLE_PUBLIC_ENTRY");
    function = f;
  }
  if (!function || function.getFunctionType().getNumResults())
    return reject("UNSUPPORTED_ENTRY_SIGNATURE");
  for (Value arg : function.getArguments())
    if (isa<RankedTensorType>(arg.getType()) || !supportedType(arg.getType()))
      return reject("UNSUPPORTED_ENTRY_SIGNATURE");
  if (function->hasAttr("hacc.block_mapping_owner")) return reject("ALREADY_MAPPED");
  if (auto attr = function->getAttr("auto_blockify_size")) {
    auto size = dyn_cast<IntegerAttr>(attr);
    if (!size || size.getInt() != 1) return reject("G0_MAPPING_CONFLICT");
  }
  auto &block = function.getBody().front();
  SmallVector<Operation *> originals;
  bool axes[3] = {};
  for (Operation &op : block) {
    originals.push_back(&op);
    auto name = op.getName().getStringRef();
    ++report.operations[name.str()];
    if (name.starts_with("scf.")) return reject("BASIC_CONTROL_FLOW_NOT_IMPLEMENTED");
    if (name.starts_with("tt.atomic") || name == "tt.print" || name == "tt.assert")
      return reject("UNSUPPORTED_EFFECT");
    if (name == "tt.call" || name == "tt.extern_elementwise") return reject("UNSUPPORTED_CALL");
    if (!supported(name)) return reject("BASIC_OP_NOT_IMPLEMENTED");
    if (!attrsSupported(&op)) return reject("UNSUPPORTED_OPERATION_ATTRIBUTE");
    for (Type t : op.getOperandTypes()) if (!supportedType(t)) return reject("UNSUPPORTED_TYPE_OR_RANK");
    for (Type t : op.getResultTypes()) {
      if (!supportedType(t)) return reject("UNSUPPORTED_TYPE_OR_RANK");
      if (auto tensor = dyn_cast<RankedTensorType>(t))
        if (tensor.getNumElements() > static_cast<int64_t>((1 << 20) / llvm::PowerOf2Ceil(request.factor)))
          return reject("TENSOR_SHAPE_BUDGET_EXCEEDED");
    }
    if (auto pid = dyn_cast<triton::GetProgramIdOp>(op)) {
      int axis = static_cast<int>(pid.getAxis());
      if (axis < 0 || axis > 2) return reject("INVALID_PROGRAM_AXIS");
      axes[axis] = true;
    }
    if (auto count = dyn_cast<triton::GetNumProgramsOp>(op)) {
      int axis = static_cast<int>(count.getAxis());
      if (axis < 0 || axis > 2) return reject("INVALID_PROGRAM_AXIS");
    }
    if (auto reduce = dyn_cast<triton::ReduceOp>(op)) {
      if (reduce.getNumResults() != 1 || reduce.getSrcs().size() != 1 ||
          reduce.getAxis() != 0 || scalar(reduce.getSrcs()[0].getType()) != Float32Type::get(module.getContext()))
        return reject("UNSUPPORTED_REDUCTION");
      auto &combiner = reduce.getCombineOp().front();
      if (combiner.getNumArguments() != 2 || combiner.getOperations().size() != 2)
        return reject("UNSUPPORTED_REDUCTION_COMBINER");
      auto add = dyn_cast<arith::AddFOp>(combiner.front());
      auto ret = dyn_cast<triton::ReduceReturnOp>(combiner.back());
      if (!add || !ret || !attrsSupported(add) || ret.getNumOperands() != 1 ||
          ret.getOperand(0) != add.getResult() ||
          !((add.getLhs() == combiner.getArgument(0) && add.getRhs() == combiner.getArgument(1)) ||
            (add.getLhs() == combiner.getArgument(1) && add.getRhs() == combiner.getArgument(0))))
        return reject("UNSUPPORTED_REDUCTION_COMBINER");
    } else if (op.getNumRegions()) return reject("BASIC_CONTROL_FLOW_NOT_IMPLEMENTED");
  }
  // The runtime guard proves every original axis. Prefer the first used axis
  // for deterministic row batching; column masks spanning [F,D] need a more
  // general non-rectangular mask lowering and can still be requested explicitly.
  int axis = request.axis;
  if (axis < 0)
    for (int i = 0; i < 3; ++i)
      if (axes[i]) { axis = i; break; }
  if (axis < 0) return reject("NO_PROGRAM_AXIS");
  if (!axes[axis]) return reject("UNUSED_BATCH_AXIS");
  AccessAnalysis analysis(function);
  for (auto *op : originals) {
    if (auto load = dyn_cast<triton::LoadOp>(op)) {
      if (load.getIsVolatile() || !load.getBoundaryCheck().empty() ||
          load.getCache() != triton::CacheModifier::NONE || load.getEvict() != triton::EvictionPolicy::NORMAL)
        return reject("UNSUPPORTED_LOAD_POLICY");
      if (load.getMask() && !load.getOther()) return reject("MASKED_LOAD_REQUIRES_OTHER");
      if (!isa<FloatType>(scalar(load.getType()))) return reject("UNSUPPORTED_LOAD_TYPE");
      analysis.add(load.getPtr(), load.getMask(), false);
    }
    if (auto store = dyn_cast<triton::StoreOp>(op)) {
      if (!store.getBoundaryCheck().empty() || store.getCache() != triton::CacheModifier::NONE ||
          store.getEvict() != triton::EvictionPolicy::NORMAL)
        return reject("UNSUPPORTED_STORE_POLICY");
      analysis.add(store.getPtr(), store.getMask(), true);
    }
    if (!analysis.reason.empty()) return reject(analysis.reason);
  }
  if (analysis.accesses.empty()) return reject("NO_MEMORY_ACCESSES");
  llvm::json::Object guard{{"version", 1}, {"accesses", std::move(analysis.accesses)}};
  report.guardJSON = llvm::formatv("{0}", J(std::move(guard))).str();
  report.selectedAxis = axis;
  report.laneWidth = llvm::PowerOf2Ceil(request.factor);
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
  for (auto *op : originals)
    if (auto pid = dyn_cast<triton::GetProgramIdOp>(op); pid && static_cast<int>(pid.getAxis()) == axis) { pidSource = op; break; }
  Value group = rewriter.clone(*pidSource)->getResult(0);
  Value group64 = rewriter.create<arith::ExtSIOp>(loc, i64, group);
  Value lane = rewriter.create<triton::MakeRangeOp>(loc, lane32, 0, lanes);
  Value laneExt = rewriter.create<arith::ExtSIOp>(loc, lane64, lane);
  Value base64 = rewriter.create<arith::MulIOp>(loc, group64, constant(i64, request.factor));
  Value baseVec = rewriter.create<triton::SplatOp>(loc, lane64, base64);
  Value p64 = rewriter.create<arith::AddIOp>(loc, baseVec, laneExt);
  Value g64 = rewriter.create<arith::ExtSIOp>(loc, i64, function.getArgument(originalArgCount + axis));
  Value gVec = rewriter.create<triton::SplatOp>(loc, lane64, g64);
  Value live = rewriter.create<arith::AndIOp>(loc,
      rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, p64, gVec),
      rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, lane, constant(lane32, request.factor)));
  // A nonempty launch guarantees group*F <= G-1 <= INT32_MAX.
  // Keep the lane range affine so normal strided memory lowering can recover
  // contiguous columns. Inactive lanes may wrap in ordinary integer ops (no
  // poison flags are admitted), but every varying memory effect is live-masked.
  Value base32 = rewriter.create<arith::TruncIOp>(loc, i32, base64);
  Value base32Vec = rewriter.create<triton::SplatOp>(loc, lane32, base32);
  Value pid = rewriter.create<arith::AddIOp>(loc, base32Vec, lane);
  IRMapping mapping;
  DenseMap<Value, bool> varying;
  for (Value arg : function.getArguments()) { mapping.map(arg, arg); varying[arg] = false; }
  auto liftedType = [&](Type t) -> RankedTensorType {
    SmallVector<int64_t> shape{lanes};
    if (auto tensor = dyn_cast<RankedTensorType>(t)) llvm::append_range(shape, tensor.getShape());
    return RankedTensorType::get(shape, scalar(t));
  };
  auto lift = [&](Value old, RankedTensorType target) -> Value {
    Value value = mapping.lookup(old);
    if (value.getType() == target) return value;
    auto tensor = dyn_cast<RankedTensorType>(value.getType());
    if (!tensor) return rewriter.create<triton::SplatOp>(loc, target, value);
    if (tensor.getRank() + 1 == target.getRank())
      value = rewriter.create<triton::ExpandDimsOp>(loc, value, varying.lookup(old) ? 1 : 0);
    if (value.getType() == target) return value;
    return rewriter.create<triton::BroadcastOp>(loc, target, value);
  };
  auto liveFor = [&](RankedTensorType target) -> Value {
    Value mask = live;
    auto maskType = RankedTensorType::get(target.getShape(), rewriter.getI1Type());
    if (target.getRank() == 2) mask = rewriter.create<triton::ExpandDimsOp>(loc, mask, 1);
    if (mask.getType() != maskType) mask = rewriter.create<triton::BroadcastOp>(loc, maskType, mask);
    return mask;
  };
  for (Operation *op : originals) {
    rewriter.setInsertionPoint(op);
    loc = op->getLoc();
    if (auto oldPid = dyn_cast<triton::GetProgramIdOp>(op); oldPid && static_cast<int>(oldPid.getAxis()) == axis) {
      mapping.map(op->getResult(0), pid); varying[op->getResult(0)] = true; continue;
    }
    if (auto programs = dyn_cast<triton::GetNumProgramsOp>(op)) {
      mapping.map(op->getResult(0), function.getArgument(originalArgCount + static_cast<unsigned>(programs.getAxis())));
      varying[op->getResult(0)] = false; continue;
    }
    bool var = llvm::any_of(op->getOperands(), [&](Value v) { return varying.lookup(v); });
    // Every store must be lane-exclusive. The original-domain host proof also
    // checks axes not batched by this request.
    if (isa<triton::StoreOp>(op) && !var) return reject("UNIFORM_STORE");
    Operation *replacement = nullptr;
    if (!var || isa<triton::ReturnOp>(op)) {
      replacement = rewriter.clone(*op, mapping);
    } else if (auto splat = dyn_cast<triton::SplatOp>(op)) {
      Value result = lift(splat.getSrc(), liftedType(splat.getType()));
      mapping.map(splat.getResult(), result); varying[splat.getResult()] = true; continue;
    } else if (auto reduce = dyn_cast<triton::ReduceOp>(op)) {
      auto result = rewriter.create<triton::ReduceOp>(loc, ValueRange{mapping.lookup(reduce.getSrcs()[0])}, 1);
      rewriter.cloneRegionBefore(reduce.getCombineOp(), result.getCombineOp(), result.getCombineOp().end());
      replacement = result;
    } else if (auto load = dyn_cast<triton::LoadOp>(op)) {
      auto type = liftedType(load.getType());
      Value ptr = lift(load.getPtr(), RankedTensorType::get(type.getShape(), scalar(load.getPtr().getType())));
      Value mask = liveFor(type);
      if (load.getMask()) mask = rewriter.create<arith::AndIOp>(loc, mask,
          lift(load.getMask(), RankedTensorType::get(type.getShape(), rewriter.getI1Type())));
      Value other = load.getOther() ? lift(load.getOther(), type) :
          rewriter.create<arith::ConstantOp>(loc, type, DenseElementsAttr::get(type, rewriter.getZeroAttr(type.getElementType())));
      replacement = rewriter.create<triton::LoadOp>(loc, ptr, mask, other,
          load.getBoundaryCheck(), load.getPadding(), load.getCache(), load.getEvict(), false);
    } else if (auto store = dyn_cast<triton::StoreOp>(op)) {
      auto type = liftedType(store.getValue().getType());
      Value ptr = lift(store.getPtr(), RankedTensorType::get(type.getShape(), scalar(store.getPtr().getType())));
      Value mask = liveFor(type);
      if (store.getMask()) mask = rewriter.create<arith::AndIOp>(loc, mask,
          lift(store.getMask(), RankedTensorType::get(type.getShape(), rewriter.getI1Type())));
      replacement = rewriter.create<triton::StoreOp>(loc, ptr, lift(store.getValue(), type), mask,
          store.getBoundaryCheck(), store.getCache(), store.getEvict());
    } else {
      IRMapping operands;
      SmallVector<int64_t> shape = llvm::to_vector(liftedType(op->getResult(0).getType()).getShape());
      for (Value operand : op->getOperands())
        operands.map(operand, lift(operand, RankedTensorType::get(shape, scalar(operand.getType()))));
      replacement = rewriter.clone(*op, operands);
      for (auto [old, result] : llvm::zip(op->getResults(), replacement->getResults()))
        result.setType(liftedType(old.getType()));
    }
    for (auto [old, result] : llvm::zip(op->getResults(), replacement->getResults())) {
      mapping.map(old, result); varying[old] = var;
    }
  }
  for (auto *op : llvm::reverse(originals)) rewriter.eraseOp(op);
  ProgramGridTransformContract contract;
  contract.version = 3;
  contract.transforms.push_back({0, axis, request.factor, false, false});
  if (failed(setProgramGridTransformContract(module, contract)))
    return module.emitError("block batching generated an invalid grid contract"), failure();
  module->setAttr("hacc.block_mapping_owner", rewriter.getStringAttr("block_batching"));
  function->setAttr("hacc.block_mapping_owner", rewriter.getStringAttr("block_batching"));
  if (failed(verify(module)))
    return module.emitError("block batching generated invalid TTIR"), failure();
  report.status = "REWRITTEN";
  report.reason = "";
  return report;
}
FailureOr<BatchRewriteReport> rewriteBlockBatching(ModuleOp module, BatchRequest request) {
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
