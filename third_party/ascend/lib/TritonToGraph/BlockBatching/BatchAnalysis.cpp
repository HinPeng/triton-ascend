// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchPlan.h"
#include "TritonToGraph/BlockBatching/ReductionModels.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>

namespace mlir::triton::cfg {
std::string describeBatchComputation(Operation *root) {
  llvm::json::Array aggregations;
  llvm::DenseMap<Operation *, int64_t> ids;
  root->walk([&](Operation *op) {
    if (isa<triton::ReduceOp, triton::ScanOp>(op))
      ids[op] = ids.size();
  });
  root->walk([&](Operation *op) {
    if (!ids.count(op))
      return;
    llvm::json::Object record;
    record["id"] = ids.lookup(op);
    record["kind"] = isa<triton::ScanOp>(op) ? "scan" : "reduce";
    auto type = op->getNumOperands()
                    ? dyn_cast<RankedTensorType>(op->getOperand(0).getType())
                    : RankedTensorType{};
    auto axis = op->getAttrOfType<IntegerAttr>("axis");
    if (type && axis && axis.getInt() >= 0 && axis.getInt() < type.getRank()) {
      llvm::json::Array shape, effective;
      int64_t effectiveAxis = 0;
      for (auto [dim, extent] : llvm::enumerate(type.getShape())) {
        shape.push_back(extent);
        if (extent != 1) {
          effective.push_back(extent);
          if (static_cast<int64_t>(dim) < axis.getInt())
            ++effectiveAxis;
        }
      }
      record["original_shape"] = std::move(shape);
      record["effective_shape"] = std::move(effective);
      record["original_axis"] = axis.getInt();
      record["lifted_axis"] = axis.getInt() + 1;
      record["axis_extent"] = type.getDimSize(axis.getInt());
      record["effective_axis"] = type.getDimSize(axis.getInt()) == 1
                                     ? llvm::json::Value(nullptr)
                                     : llvm::json::Value(effectiveAxis);
      record["domain"] =
          isa<triton::ReduceOp>(op)
              ? "original_program_reduction"
              : (hasBatchIndependentScanDomain(type, axis.getInt())
                     ? "independent_row_scan"
                     : "unsupported_multidimensional_scan");
    } else {
      record["domain"] = "unproven";
    }
    record["arity"] = static_cast<int64_t>(op->getNumOperands());
    int64_t loops = 0, branches = 0;
    for (Operation *parent = op->getParentOp(); parent && parent != root;
         parent = parent->getParentOp()) {
      loops += isa<scf::ForOp>(parent);
      branches += isa<scf::IfOp>(parent);
    }
    record["loop_depth"] = loops;
    record["conditional_depth"] = branches;
    // Immediate SSA producers only; this is not a transitive dependence proof.
    llvm::json::Array producers;
    for (Value input : op->getOperands()) {
      auto found = ids.find(input.getDefiningOp());
      producers.push_back(found == ids.end()
                              ? llvm::json::Value(nullptr)
                              : llvm::json::Value(found->second));
    }
    record["direct_input_aggregations"] = std::move(producers);
    aggregations.push_back(std::move(record));
  });
  llvm::json::Object result;
  result["version"] = 1;
  result["scope"] = "normalized_original_program";
  result["lane_policy"] = "separate_original_domains";
  result["aggregations"] = std::move(aggregations);
  return llvm::formatv("{0}", llvm::json::Value(std::move(result))).str();
}

void setBatchDiagnostic(BatchAnalysisReport &report, Operation *op,
                        llvm::StringRef reason) {
  report.reason = reason.str();
  report.operation = op->getName().getStringRef().str();
  report.context = "BlockBody";
  Operation *child = op;
  for (Operation *parent = op->getParentOp(); parent;
       child = parent, parent = parent->getParentOp()) {
    if (parent->getName().getStringRef() == "tt.reduce") {
      report.context = "ReduceCombiner";
      break;
    }
    if (parent->getName().getStringRef() == "tt.scan") {
      report.context = "ScanCombiner";
      break;
    }
    if (parent->getName().getStringRef() == "scf.for") {
      report.context = "ForBody";
      break;
    }
    if (parent->getName().getStringRef() == "scf.if") {
      report.context = child->getParentRegion() == &parent->getRegion(0)
                           ? "IfThen"
                           : "IfElse";
      break;
    }
  }
  report.location.clear();
  llvm::raw_string_ostream stream(report.location);
  op->getLoc().print(stream);
}

FailureOr<BatchAnalysisReport> analyzeBlockBatching(ModuleOp module,
                                                    BatchRequest request) {
  if (!request.isValid())
    return failure();
  BatchAnalysisReport report;
  if (request.factor == 1) {
    report.status = "DISABLED";
    report.reason = "DISABLED";
    return report;
  }
  if (request.factor > std::numeric_limits<int32_t>::max()) {
    report.reason = "FACTOR_OUT_OF_RANGE";
    return report;
  }
  unsigned entries = 0;
  for (auto function : module.getOps<triton::FuncOp>())
    if (function.isPublic())
      ++entries;
  if (entries != 1) {
    report.reason = "EXPECTED_SINGLE_PUBLIC_ENTRY";
    return report;
  }
  module.walk([&](Operation *op) {
    const auto name = op->getName().getStringRef();
    ++report.operations[name.str()];
  });
  if (report.reason.empty()) {
    // Analyze with the same registry and verifier on a private clone. This
    // report never authorizes a launch: alias/resource/mode proofs remain.
    request.semanticOnly = true;
    auto rewrite = probeBlockBatching(module, request);
    if (failed(rewrite))
      return failure();
    report.status =
        rewrite->status == "REWRITTEN" ? "ANALYZED" : rewrite->status;
    report.reason = rewrite->reason;
    report.location = rewrite->location;
    report.operation = rewrite->operation;
    report.context = rewrite->context;
    report.layoutJSON = rewrite->layoutJSON;
    report.computationJSON = rewrite->computationJSON;
  }
  return report;
}
} // namespace mlir::triton::cfg
