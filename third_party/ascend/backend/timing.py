"""Compile-time profiling for triton-ascend compilation pipeline.

Controlled by env var TRITON_ENABLE_COMPILE_TIMING=1.
Output to $PWD/.triton_timing/ or stdout if TRITON_TIMING_STDOUT=1.
"""

import io
import json
import os
import re
import time
import contextlib
from pathlib import Path


# Known pass names in ttir_to_linalg order, used to match MLIR timing output
_PASS_NAMES = [
    "auto_blockify",
    "dag_sync",
    "dag_scope",
    "dag_ssbuffer",
    "triton_control_flow_opt",
    "triton_to_structure",
    "discrete_mask_access_conversion",
    "triton_to_annotation",
    "triton_to_unstructure",
    "triton_to_hivm",
    "triton_to_hfusion",
    "triton_to_llvm",
    "bubble_up_operation",
    "triton_to_linalg",
    "dynamic_cv_pipeline",
]

# Pass complexity annotations (static analysis from C++ source)
_PASS_COMPLEXITY = {
    "auto_blockify":                    ("O(n)",    "linear scan per block"),
    "dag_sync":                         ("O(n^2)",  "dependency graph, pairwise op analysis"),
    "dag_scope":                        ("O(n^2)",  "scope partitioning on dependency graph"),
    "dag_ssbuffer":                     ("O(n^2)",  "live-range conflict resolution for SSA buffers"),
    "triton_control_flow_opt":           ("O(n*d)",  "CFG optimization, d=nesting depth"),
    "triton_to_structure":              ("O(n*r)",  "structural conversion, r=region count"),
    "discrete_mask_access_conversion":  ("O(n)",    "linear scan of memory access ops"),
    "triton_to_annotation":             ("O(n)",    "linear attribute annotation"),
    "triton_to_unstructure":            ("O(n)",    "per-op unstructuring"),
    "triton_to_hivm":                   ("O(n*t)",  "dialect lowering, t=type variants"),
    "triton_to_hfusion":                ("O(n^2)",  "horizontal fusion: pairwise producer-consumer matching"),
    "triton_to_llvm":                   ("O(n)",    "per-op LLVM dialect lowering"),
    "bubble_up_operation":              ("O(n*d)",  "cross-region op bubbling, d=nesting depth"),
    "triton_to_linalg":                 ("O(n)",    "per-op Linalg lowering"),
    "dynamic_cv_pipeline":              ("O(n^2)",  "global scheduling analysis for CV pipeline"),
}


def _timing_enabled():
    return os.getenv("TRITON_ENABLE_COMPILE_TIMING", "0") in ("1", "true", "True")


def _output_to_stdout():
    return os.getenv("TRITON_TIMING_STDOUT", "0") in ("1", "true", "True")


class _CompileTimingCollector:
    """Singleton collector for per-kernel compile timing data."""

    def __init__(self):
        self._records = []

    def add_record(self, record: dict):
        self._records.append(record)

    def flush(self):
        if not self._records:
            return
        if _output_to_stdout():
            for r in self._records:
                print(json.dumps(r))
        else:
            outdir = Path.cwd() / ".triton_timing"
            outdir.mkdir(exist_ok=True)
            for r in self._records:
                fname = f"{r['kernel_name']}_{r['hash'][:8]}.json"
                (outdir / fname).write_text(json.dumps(r, indent=2))
        self._records.clear()


_collector = _CompileTimingCollector()


def record_kernel_timing(kernel_name: str, kernel_hash: str, phases: dict,
                         ir_stats: dict = None):
    """Record compile timing for one kernel. Called when all phases are complete."""
    if not _timing_enabled():
        return
    import datetime
    record = {
        "kernel_name": kernel_name,
        "hash": kernel_hash,
        "timestamp": datetime.datetime.now().isoformat(),
        "ir_stats": ir_stats or {},
        "phases": phases,
    }
    _collector.add_record(record)


class time_phase:
    """Context manager to time a compilation phase.

    Usage:
        with time_phase("make_ttir") as ctx:
            ...
        # ctx.elapsed_ms is available after the block
    """

    def __init__(self, phase_name: str):
        self.phase_name = phase_name
        self.elapsed_ms = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000.0


def parse_mlir_timing(mlir_timing_output: str) -> list:
    """Parse MLIR PassTimingDisplay stderr output into structured pass timing list.

    Expected input format:
        ===-----------------------------------------------------------===
                             ... Pass execution timing report ...
        ===-----------------------------------------------------------===
          Total Execution Time: 0.1503 seconds

          ---Wall Time---  ---Name---
          0.0032 (  2.1%)  AutoBlockifyPass
          0.0081 (  5.4%)  TritonControlFlowOptPass
          ...
          0.1503 (100.0%)  Total

    Returns list of dicts with keys: name, elapsed_ms, complexity, note
    """
    if not mlir_timing_output:
        return []

    # MLIR timing output uses pass class names (CamelCase + "Pass" suffix).
    # Map common MLIR display names back to our canonical pass names.
    _NAME_MAP = {
        "autoblockifypass":                    "auto_blockify",
        "dagsyncpass":                        "dag_sync",
        "dagscopepass":                       "dag_scope",
        "dagssbufferpass":                     "dag_ssbuffer",
        "tritoncontrolflowoptpass":            "triton_control_flow_opt",
        "tritontostructuredpass":             "triton_to_structure",
        "discretemaskaccessconversionpass":    "discrete_mask_access_conversion",
        "tritontoannotationpass":             "triton_to_annotation",
        "tritontounstructurepass":            "triton_to_unstructure",
        "tritontohivmpass":                   "triton_to_hivm",
        "tritontohfusionpass":                "triton_to_hfusion",
        "tritontollvmpass":                   "triton_to_llvm",
        "bubbleupoperationpass":              "bubble_up_operation",
        "tritontolinalgpass":                 "triton_to_linalg",
        "dynamiccvpipelinepass":              "dynamic_cv_pipeline",
        "canonicalizer":                      "canonicalizer",
        "cse":                                "cse",
    }

    results = []
    # Match lines like: "  0.0032 (  2.1%)  PassName"
    pattern = re.compile(r"^\s*([\d.]+)\s+\(\s*[\d.]+\%\)\s+(.+)")

    for line in mlir_timing_output.splitlines():
        m = pattern.match(line.strip())
        if m:
            elapsed_s = float(m.group(1))
            pass_name = m.group(2).strip()
            if pass_name == "Total":
                continue
            normalized = _NAME_MAP.get(pass_name.lower().replace(" ", ""), pass_name)
            complexity, note = _PASS_COMPLEXITY.get(normalized, ("?", ""))
            results.append({
                "name": normalized,
                "elapsed_ms": round(elapsed_s * 1000.0, 4),
                "complexity": complexity,
                "note": note,
            })

    return results


def count_ir_ops(mod) -> int:
    """Count operations in an MLIR module (approximate via line count)."""
    try:
        return str(mod).count("\n")
    except Exception:
        return -1
