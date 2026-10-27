# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Finalize a candidate using the ordinary TA/backend compilation contract.

Call only after successful compilation and verification of the output binary.
UB numbers are optional diagnostics, not another resource admission gate.
No custom AscendNPU-IR report or transitive SIMD proof is required.
"""
from .block_batching import BatchingRejected


VALIDATION_POLICY = "ta_only_v1"


def validate_compilation_result(metadata):
    # Reuse the normal launcher's callback/default-zero workspace contract.
    # Explicit invalid values must not silently become the default.
    workspace = metadata.get("workspace_size", 0)
    if type(workspace) is not int or workspace < 0:
        raise BatchingRejected("INVALID_WORKSPACE_SIZE")
    if metadata.get("is_pure_simt") or metadata.get("mix_mode") != "aiv":
        raise BatchingRejected("UNSUPPORTED_FINAL_EXECUTION_MODE")
    metadata["block_batching"].update(
        compile_status="COMPILED_GUARDED", owner="block_batching",
        validation_policy=VALIDATION_POLICY,
        resource_validation="backend_compile_success",
        execution_mode_validation="standard_backend_metadata",
        internal_simt="not_inspected",
        workspace_bytes=workspace,
        workspace_source="callback" if "workspace_size" in metadata else "standard_backend_default",
    )
