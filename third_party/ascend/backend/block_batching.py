# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Block batching request and analysis contracts.

Candidates require a proved runtime memory domain, a matching native ABI and
a successfully compiled backend artifact under the ordinary launch contract. Baseline and candidate are separate complete
compiler artifacts; rejected or guarded calls select the complete baseline.
"""
from dataclasses import dataclass

IMPLEMENTATION_VERSION = 15


@dataclass(frozen=True)
class BatchRequest:
    factor: int = 1
    axis: int = -1

    def __post_init__(self):
        if type(self.factor) is not int or self.factor <= 0:
            raise ValueError("block_batching_factor must be a positive integer (not bool)")
        if type(self.axis) is not int or self.axis not in (-1, 0, 1, 2):
            raise ValueError("block_batching_axis must be -1, 0, 1 or 2 (not bool)")


def analyze_request(mod, request, binding):
    """Read-only preflight. Never suppress legacy optimizations on the baseline."""
    if request.factor == 1:
        return {
            "report_version": 1, "compile_status": "DISABLED", "requested_factor": 1, "requested_axis": request.axis,
            "applied_factor": 1, "owner": "none", "reason": "DISABLED"
        }
    # MLIR shapes and the future native launch contract use signed i32 extents.
    # A positive Python integer outside that capability is not a config error.
    if request.factor > 2147483647:
        return {
            "report_version": 1, "compile_status": "REJECTED", "requested_factor": request.factor, "requested_axis":
            request.axis, "applied_factor": 1, "owner": "none", "reason": "FACTOR_OUT_OF_RANGE"
        }
    analyze = getattr(binding, "analyze_block_batching", None)
    if analyze is None:
        raise RuntimeError("block_batching requires a matching rebuilt Ascend C++ extension")
    return dict(analyze(mod, request.factor, request.axis))


class BatchingRejected(RuntimeError):
    """A modeled capability/proof rejection, not an internal compiler error."""


class LayoutTargetRejected(BatchingRejected):
    """A resource refusal of an actually rewritten layout at the NPUIR stage."""

    def __init__(self, factor, axis, diagnostic):
        super().__init__("LAYOUT_TARGET_RESOURCE_REJECTED")
        self.factor, self.axis = factor, axis
        self.diagnostic = diagnostic
        self.stage = "npubin"


def _internal_backend_error(returncode, diagnostic):
    return returncode < 0 or returncode in (134, 139) or any(
        marker in diagnostic for marker in ("PLEASE submit a bug report", "LLVM ERROR:", "Assertion", "Sanitizer",
                                            "runtime error:", "failed to verify", "verification failed"))


def _layout_resource_refusal(returncode, diagnostic):
    import re
    if _internal_backend_error(returncode, diagnostic):
        return False
    # Match the standard backend's resource diagnostic, not an arbitrary
    # occurrence of "UB" and "overflow" in an internal compiler failure.
    return re.search(r"\bub overflow,\s*requires\s+\d+\s+bits\s+while\s+\d+\s+bits\s+available\b", diagnostic,
                     re.IGNORECASE) is not None


def _compile_layout_candidate(compile_variant, options, frozen, request):
    from triton.compiler.errors import MLIRCompilationError
    policy = options.get("block_batching_layout", "auto")
    try:
        return compile_variant(dict(options), frozen), None
    except (LayoutTargetRejected, MLIRCompilationError) as error:
        cause = error
        while isinstance(cause, MLIRCompilationError) and cause.__cause__ is not None:
            cause = cause.__cause__
        if not isinstance(cause, LayoutTargetRejected):
            raise
        if (policy != "auto" or cause.factor != request.factor or cause.axis != request.axis
                or cause.stage != "npubin"):
            raise
        # Re-enter compilation from the SAME frozen frontend source. A distinct
        # option identity prevents reusing auto IR, metadata or failed attempts.
        fallback = dict(options, block_batching_layout="semantic_only")
        detail = {"reason": str(cause), "stage": cause.stage, "diagnostic": cause.diagnostic}
        import subprocess
        try:
            candidate = compile_variant(fallback, frozen)
        except (BatchingRejected, subprocess.CalledProcessError, MLIRCompilationError) as fallback_error:
            fallback_error.block_batching_layout_fallback = detail
            raise
        return candidate, detail


def _runtime_positions(src):
    constants = getattr(src, "constants", {})
    result = []
    for index, name in enumerate(src.fn.arg_names):
        signature = src.signature.get(name, src.signature.get(index))
        if not isinstance(signature, str):
            raise BatchingRejected("NESTED_SIGNATURE_NOT_SUPPORTED")
        if signature != "constexpr" and (index, ) not in constants and index not in constants and name not in constants:
            result.append(index)
    return result


def _capture_state():
    import sys
    torch = sys.modules.get("torch")
    query = getattr(getattr(torch, "npu", None), "is_current_stream_capturing", None)
    return query() if query is not None else None


class BatchKernel:
    """CompiledKernel-compatible facade selecting a complete variant per call."""

    def __init__(self, baseline, candidate, request, report, positions):
        import hashlib
        import json
        import threading
        from collections import namedtuple
        self.baseline, self.candidate = baseline, candidate
        self.request, self.block_batching = request, report
        self.positions = positions
        from .block_batching_guard import RuntimeGuard
        self._guard = RuntimeGuard(report["guard"], request.factor) if candidate is not None else None
        self._primary = candidate if candidate is not None else baseline
        self._lock = threading.RLock()
        self._ready = False
        self._local = threading.local()
        self.hash = candidate.hash if candidate is not None else hashlib.sha256(
            f"{baseline.hash}:block_batching:{request.factor}:{request.axis}:{IMPLEMENTATION_VERSION}".encode(
            )).hexdigest()
        fields = dict(self._primary.metadata._asdict(), block_batching=report)
        self.metadata = namedtuple("BatchKernelMetadata", sorted(fields))(**fields)
        from triton.runtime.cache import get_cache_manager
        manifest = {
            "version": 1, "baseline_hash": baseline.hash, "candidate_hash":
            candidate.hash if candidate is not None else None, "request":
            {"factor": request.factor, "axis": request.axis}, "report": report
        }
        get_cache_manager(self.hash).put(json.dumps(manifest, sort_keys=True), "block_batching.bundle.json")

    def __getattr__(self, name):
        return getattr(self._primary, name)

    @property
    def last_launch(self):
        return getattr(self._local, "launch", {"runtime_status": "NOT_LAUNCHED"})

    def _init_handles(self):
        with self._lock:
            if self._ready:
                return
            # Capture cannot trigger loading a new guarded variant. A normal
            # eager warmup initializes both complete handles beforehand.
            if _capture_state() and getattr(self.baseline, "_run", None) is None:
                raise RuntimeError("warm up the block batching baseline before graph capture")
            self.baseline._init_handles()
            if self.candidate is not None:
                from triton.runtime.errors import OutOfResources
                try:
                    self.candidate._init_handles()
                except (BatchingRejected, OutOfResources) as error:
                    self.block_batching.update(compile_status="REJECTED", reason=str(error), owner="none")
                    self.candidate = None
                    self._primary = self.baseline
                    from collections import namedtuple
                    fields = dict(self.baseline.metadata._asdict(), block_batching=self.block_batching)
                    self.metadata = namedtuple("BatchKernelMetadata", sorted(fields))(**fields)
            self._ready = True

    @property
    def run(self):
        self._init_handles()
        return self._dispatch

    def launch_metadata(self, grid, stream, *arguments):
        # Hooks are constructed in _dispatch, after actual variant selection.
        return None

    def _dispatch(self, *launch):
        from triton import knobs
        from .block_batching_guard import integer, I32_MAX
        grid = tuple(integer(x) for x in launch[:3])
        if any(x < 0 or x > I32_MAX for x in grid):
            raise ValueError("original grid must contain nonnegative i32 extents")
        if any(x == 0 for x in grid):
            self._local.launch = {"runtime_status": "EMPTY_GRID", "applied_factor": 1, "original_grid": grid}
            return
        selected = self.baseline
        reason = self.block_batching.get("reason", "CANDIDATE_UNAVAILABLE")
        accepted = False
        arguments = launch[9:]
        if self.candidate is not None:
            capture = _capture_state()
            if capture is not False:
                reason = "CAPTURE_BASELINE" if capture else "CAPTURE_QUERY_UNAVAILABLE"
            else:
                runtime_args = [arguments[index] for index in self.positions]
                accepted, reason = self._guard(runtime_args, grid)
                if accepted:
                    selected = self.candidate
        # Never mix baseline function/metadata/workspace/layout with candidate.
        forwarded = list(launch)
        forwarded[4], forwarded[5] = selected.function, selected.packed_metadata
        forwarded[6] = selected.launch_metadata(grid, launch[3], *arguments)
        result = selected.run(*forwarded)
        factor = self.request.factor if accepted else 1
        batch_grid = list(grid)
        if accepted:
            axis = self.block_batching["selected_axis"]
            batch_grid[axis] = (grid[axis] + factor - 1) // factor
        self._local.launch = {
            "runtime_status": "APPLIED" if accepted else "BASELINE_SELECTED", "owner":
            "block_batching" if accepted else "baseline", "requested_factor": self.request.factor, "applied_factor":
            factor, "selected_axis": self.block_batching.get("selected_axis"), "original_grid": grid, "batch_grid":
            tuple(batch_grid), "reason": reason, "plan_id": self.hash
        }
        if knobs.ascend.compile_only:
            self._local.launch.update(runtime_status="NOT_LAUNCHED", reason="COMPILE_ONLY")
        return result

    def __getitem__(self, grid):

        def runner(*args, stream=None):
            from triton import knobs
            from triton.runtime import driver
            self._init_handles()
            if stream is None:
                stream = driver.active.get_current_stream(driver.active.get_current_device())
            return self.run(*grid, stream, self.function, self.packed_metadata, None, knobs.runtime.launch_enter_hook,
                            knobs.runtime.launch_exit_hook, *args)

        return runner


def _freeze_source(backend, src, baseline_options, backend_revision):
    """Generate frontend IR once, then parse the same immutable snapshot per variant."""
    import hashlib
    from pathlib import Path
    from triton._C.libtriton import ir
    from triton.compiler.compiler import ASTSource
    from triton.runtime.cache import get_cache_manager
    if not isinstance(src, ASTSource):
        return src
    options = backend.parse_options(baseline_options)
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = src.make_ir(backend.target, options, backend.get_codegen_implementation(options), backend.get_module_map(),
                         context)
    text = str(module)
    fingerprint = hashlib.sha256((src.hash() + "\n" + backend_revision + "\n" + text).encode()).hexdigest()
    from triton.runtime.cache import FileCacheManager
    cache = get_cache_manager(fingerprint)
    if type(cache) is not FileCacheManager:
        raise BatchingRejected("CUSTOM_CACHE_NOT_SUPPORTED")
    path = cache.put(text, "block_batching.original.ttir", binary=False)

    class FrozenSource(ASTSource):

        def __init__(self):
            self.__dict__.update(src.__dict__)
            self.block_batching_source_hash = fingerprint

        def hash(self):
            return fingerprint

        def make_ir(self, target, options, codegen_fns, module_map, context):
            parsed = ir.parse_mlir_module(str(Path(path)), context)
            parsed.context = context
            return parsed

    return FrozenSource()


def compile_bundle(backend, src, options, compile_variant):
    import subprocess
    import hashlib
    from triton.compiler.errors import MLIRCompilationError
    request = BatchRequest(options.get("block_batching_factor", 1), options.get("block_batching_axis", -1))
    normalized = backend.parse_options(options)
    baseline_options = dict(options, block_batching_factor=1, block_batching_axis=-1)
    freeze_rejection = None
    try:
        frozen = _freeze_source(backend, src, baseline_options, normalized.block_batching_backend_revision)
    except BatchingRejected as error:
        frozen, freeze_rejection = src, str(error)
    baseline = compile_variant(baseline_options, frozen)
    report = {
        "report_version": 1, "compile_status": "REJECTED", "owner": "none", "requested_factor": request.factor,
        "requested_axis": request.axis, "applied_factor": 1
    }
    candidate, positions = None, []
    try:
        if freeze_rejection:
            raise BatchingRejected(freeze_rejection)
        if not str(backend.target.arch).startswith(("Ascend950", "Ascend910_95")):
            raise BatchingRejected("TARGET_NOT_VALIDATED")
        if normalized.is_pure_simt or normalized.compile_mode == "simt_only":
            raise BatchingRejected("SIMT_NOT_SUPPORTED")
        from .utils import backend_policy
        if backend_policy not in (None, "torch_npu"):
            raise BatchingRejected("FRAMEWORK_NOT_VALIDATED")
        if not hasattr(src, "fn"):
            raise BatchingRejected("AST_SOURCE_REQUIRED")
        positions = _runtime_positions(src)
        candidate, layout_fallback = _compile_layout_candidate(compile_variant, options, frozen, request)
        from .block_batching_report import VALIDATION_POLICY
        native_report = getattr(candidate.metadata, "block_batching", {})
        if (native_report.get("compile_status") != "COMPILED_GUARDED"
                or native_report.get("validation_policy") != VALIDATION_POLICY):
            raise BatchingRejected("CANDIDATE_VALIDATION_MISSING")
        if (native_report.get("implementation_version") != IMPLEMENTATION_VERSION
                or native_report.get("lane_width") != request.factor):
            raise BatchingRejected("CANDIDATE_LAYOUT_MISMATCH")
        expected = getattr(candidate.metadata, "block_batching_binary_sha256", None)
        if expected != hashlib.sha256(candidate.kernel).hexdigest():
            raise BatchingRejected("CANDIDATE_BINARY_CHANGED")
        report = dict(native_report)
        if layout_fallback:
            report["layout"] = dict(report["layout"], requested_policy="auto", selected_policy="semantic_only",
                                    compile_fallback=layout_fallback)
    except (BatchingRejected, subprocess.CalledProcessError, MLIRCompilationError) as error:
        if hasattr(error, "block_batching_layout_fallback"):
            report["layout"] = {
                "requested_policy": "auto", "selected_policy": "baseline", "compile_fallback":
                error.block_batching_layout_fallback
            }
        cause = error
        while isinstance(cause, MLIRCompilationError) and cause.__cause__ is not None:
            cause = cause.__cause__
        if isinstance(cause, BatchingRejected):
            report["reason"] = str(cause)
        elif isinstance(cause, subprocess.CalledProcessError):
            detail = cause.stderr.decode(
                errors="replace") if isinstance(cause.stderr, bytes) else str(cause.stderr or "")
            if _internal_backend_error(cause.returncode, detail):
                raise
            report.update(reason="BACKEND_COMPILE_REJECTED", backend_returncode=cause.returncode,
                          backend_diagnostic=detail[-16000:])
        else:
            # Internal verifier/runtime errors and unknown compiler failures
            # retain their original wrapped diagnostics instead of disappearing
            # behind a successful baseline launch.
            raise
        candidate = None
    return BatchKernel(baseline, candidate, request, report, positions)
