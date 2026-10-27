// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchPlan.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>

namespace mlir::triton::cfg {
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
    if (!report.reason.empty())
      return;
    if (name.starts_with("scf."))
      report.reason = "BASIC_CONTROL_FLOW_NOT_IMPLEMENTED";
    else if (name == "tt.call" || name == "tt.extern_elementwise")
      report.reason = "UNSUPPORTED_CALL";
    else if (name.starts_with("tt.atomic") || name == "tt.print" ||
             name == "tt.assert")
      report.reason = "UNSUPPORTED_EFFECT";
    if (!report.reason.empty()) {
      llvm::raw_string_ostream stream(report.location);
      op->getLoc().print(stream);
    }
  });
  if (report.reason.empty()) {
    // Analyze with the same registry and verifier on a private clone. This
    // report never authorizes a launch: alias/resource/mode proofs remain.
    OwningOpRef<ModuleOp> sandbox(module.clone());
    auto rewrite = rewriteBlockBatching(sandbox.get(), request);
    if (failed(rewrite))
      return failure();
    report.status = rewrite->status == "REWRITTEN" ? "ANALYZED" : rewrite->status;
    report.reason = rewrite->reason;
  }
  return report;
}
} // namespace mlir::triton::cfg
