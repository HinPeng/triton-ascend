"""Opt-in hardware acceptance, enabled with RUN_NPU_REUSE_TESTS=1."""
import inspect
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import triton
import triton.language as tl


@triton.jit
def pointwise(a, b, out, scale, length, work, tile: tl.constexpr):
    p = tl.program_id(0)
    step = tl.num_programs(0)
    for block in range(p, work, step):
        index = block * tile + tl.arange(0, tile)
        mask = index < length
        x = tl.load(a + index, mask=mask)
        y = tl.load(b + index, mask=mask)
        result = x * scale + y
        tl.store(out + index, result, mask=mask)


GLOBAL_BIAS = tl.constexpr(3)


@triton.jit
def direct_grid_tile_kernel(a, out, length, tile: tl.constexpr, TAG: tl.constexpr):
    index = tl.program_id(0) * tile + tl.arange(0, tile)
    valid = index < length
    x = tl.load(a + index, mask=valid, other=0.0)
    tl.store(out + index, x * 2.0 + index, mask=valid)


@triton.jit
def local_index_helper(start, size: tl.constexpr, length):
    index = tl.arange(0, size) + (start + 0)
    return index, length > index


@triton.jit
def generalized_local_kernel(a, out, length, tile: tl.constexpr, ADD: tl.constexpr):
    for block in range((length + tile - 1) // tile):
        index, valid = local_index_helper(block * tile, size=tile, length=length)
        x = tl.load(a + index, mask=valid, other=0.0)
        if ADD:
            result = x + 1.0
        else:
            result = x - 1.0
        tl.store(out + index, result, mask=valid)


@triton.jit
def global_kernel(out, N: tl.constexpr):
    i = tl.program_id(0)
    if i < N:
        tl.store(out + i, N + GLOBAL_BIAS)


@triton.jit
def compile_failure_probe(out, N: tl.constexpr):
    tl.store(out, N)


@triton.jit
def arithmetic_width_probe(out, N: tl.constexpr):
    tl.store(out, N + 1)


@triton.jit
def floating_scale_probe(x, out, SCALE: tl.constexpr):
    i = tl.program_id(0)
    tl.store(out + i, tl.load(x + i) * SCALE)


@triton.jit
def constant_identity_probe(out, N: tl.constexpr, MARKER: tl.constexpr):
    if MARKER == MARKER:  # noqa: PLR0124 - exercise retained constant type/bit identity
        tl.store(out, N)


@pytest.fixture
def npu():
    if os.environ.get("RUN_NPU_REUSE_TESTS") != "1":
        pytest.skip("set RUN_NPU_REUSE_TESTS=1 on the declared NPU matrix")
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(0)
    return torch


def test_generalized_indices_helpers_and_static_paths(npu):
    kernel = triton.jit(generalized_local_kernel.fn)
    x = npu.arange(67, device="npu", dtype=npu.float32)
    out = npu.empty_like(x)
    first = kernel[(1, )](x, out, 67, 16, True)
    second = kernel[(1, )](x, out, 67, 32, True)
    npu.npu.synchronize()
    assert first.hash == second.hash
    assert npu.equal(out, x + 1)
    subtract = kernel[(1, )](x, out, 67, 32, False)
    npu.npu.synchronize()
    assert subtract.hash != first.hash
    assert npu.equal(out, x - 1)
    again = kernel[(1, )](x, out, 67, 64, True)
    npu.npu.synchronize()
    assert again.hash == first.hash and npu.equal(out, x + 1)


@pytest.mark.parametrize("mode", ["simd", "simd_simt_template"])
@pytest.mark.parametrize("old,requested", [(16, 32), (32, 16)])
@pytest.mark.parametrize("grid_kind", ["tuple", "list", "callable"])
def test_direct_grid_tile_reuses_binary_with_adjusted_grid(npu, mode, old, requested, grid_kind):
    kernel = triton.jit(direct_grid_tile_kernel.fn)
    x = npu.arange(67, device="npu", dtype=npu.float32)
    out = npu.empty_like(x)
    calls = []
    tag = f"{mode}:{old}:{grid_kind}"

    def dynamic_grid(meta):
        calls.append(meta["tile"])
        return (triton.cdiv(67, meta["tile"]), )

    def launch(tile):
        grid = dynamic_grid if grid_kind == "callable" else (triton.cdiv(67, tile), )
        if grid_kind == "list":
            grid = list(grid)
        return kernel[grid](x, out, 67, tile, tag, compile_mode=mode)

    first = launch(old)
    npu.npu.synchronize()
    out.fill_(float("nan"))
    second = launch(requested)
    npu.npu.synchronize()
    assert first.hash == second.hash and second.src.constants[(3, )] == old
    assert npu.equal(out, x * 3.0)
    assert calls == ([old, requested] if grid_kind == "callable" else [])


@pytest.mark.parametrize("grid_size", [1, 3, 7])
@pytest.mark.parametrize("grid_kind", ["tuple", "list", "callable"])
def test_tile_reuses_complete_original_binary(npu, grid_size, grid_kind):
    from triton.backends.ascend.reuse.bridge import counters
    torch = npu
    a = torch.arange(5000, device="npu", dtype=torch.float32) / 4
    b = torch.ones_like(a)
    out = torch.empty_like(a)
    calls = []

    def dynamic_grid(meta):
        calls.append((meta["tile"], meta["work"]))
        return (grid_size, )

    grid = dynamic_grid if grid_kind == "callable" else (grid_size, )
    if grid_kind == "list":
        grid = list(grid)
    first = pointwise[grid](a, b, out, 0.5, 5000, 5, 1024)
    torch.npu.synchronize()
    second = pointwise[grid](a, b, out, 0.5, 5000, 3, 2048)
    torch.npu.synchronize()
    assert first.hash == second.hash
    assert second.src.constants[(6, )] == 1024
    assert second.src.fn is pointwise  # T-only does not construct a transformed JIT
    assert torch.equal(out, a * 0.5 + b)
    assert counters()["ready_compatible_hit"] >= 1
    assert counters()["ready_registered"] >= 1
    assert calls == ([(1024, 5), (2048, 3)] if grid_kind == "callable" else [])


def test_grid_stride_callable_preserves_partial_coverage(npu):
    x = npu.arange(5000, device="npu", dtype=npu.float32) / 4
    y = npu.ones_like(x)
    out = npu.empty_like(x)
    calls = []

    def grid(meta):
        calls.append((meta["tile"], meta["work"]))
        return (3, )

    pointwise[grid](x, y, out, 0.5, 5000, 5, 1024)
    npu.npu.synchronize()
    out.fill_(-777)
    selected = pointwise[grid](x, y, out, 0.5, 5000, 2, 2048)
    npu.npu.synchronize()
    assert selected.src.constants[(6, )] == 2048
    assert npu.equal(out[:4096], x[:4096] * 0.5 + y[:4096])
    assert npu.equal(out[4096:], npu.full_like(out[4096:], -777))
    assert calls == [(1024, 5), (2048, 2)]


def test_grid_stride_callable_error_does_not_launch(npu):
    x = npu.ones(5000, device="npu")
    out = npu.empty_like(x)
    pointwise[(3, )](x, x, out, 0.5, 5000, 5, 1024)
    npu.npu.synchronize()
    out.fill_(-777)
    calls = []
    error = RuntimeError("grid callback failed")

    def grid(meta):
        calls.append((meta["tile"], meta["work"]))
        raise error

    with pytest.raises(RuntimeError) as caught:
        pointwise[grid](x, x, out, 0.5, 5000, 3, 2048)
    npu.npu.synchronize()
    assert caught.value is error and calls == [(2048, 3)]
    assert npu.equal(out, npu.full_like(out, -777))


@pytest.mark.parametrize("mode", ["simd", "simd_simt_template"])
@pytest.mark.parametrize("dtype_name", ["float16", "float32"])
def test_layernorm_internal_rblock_reuses_with_callable_grid(npu, mode, dtype_name):
    import runpy
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "autotune_ut/03-layer-norm.py"
    original = runpy.run_path(str(path))["_layer_norm_fwd_fused"].fn
    kernel = triton.jit(original.fn)
    rows, columns = 5, 67
    dtype = getattr(npu, dtype_name)
    x = -2.3 + 0.5 * npu.randn((rows, columns), dtype=dtype, device="npu")
    weight = npu.rand(columns, dtype=dtype, device="npu")
    bias = npu.rand(columns, dtype=dtype, device="npu")
    output = npu.empty_like(x)
    mean = npu.empty(rows, dtype=npu.float32, device="npu")
    rstd = npu.empty_like(mean)
    calls = []

    def grid(meta):
        calls.append(meta["RBLOCK_SIZE"])
        return (triton.cdiv(rows, meta["XBLOCK_SIZE"]), 1, 1)

    def launch(tile):
        return kernel[grid](x, output, weight, bias, mean, rstd, columns, columns, rows, 1e-5, XBLOCK_SIZE=2,
                            RBLOCK_SIZE=tile, compile_mode=mode)

    first = launch(16)
    npu.npu.synchronize()
    second = launch(32)
    npu.npu.synchronize()
    assert first.hash == second.hash and second.src.constants[(11, )] == 16
    assert calls == [16, 32]
    expected = npu.nn.functional.layer_norm(x, (columns, ), weight, bias, 1e-5)
    npu.testing.assert_close(output, expected, atol=1e-2, rtol=0)


@pytest.mark.parametrize("mode", ["simd", "simd_simt_template", "simt_only"])
def test_integer_dynamicization_across_backend_modes(npu, mode):
    from triton.compiler import make_backend
    from triton.runtime import driver
    backend = make_backend(driver.active.get_current_target())
    try:
        backend.parse_options({"compile_mode": mode})
    except ValueError as error:
        pytest.skip(str(error))  # e.g. pure SIMT on a target that does not support it
    kernel = triton.jit(compile_failure_probe.fn)
    out = npu.zeros(1, device="npu", dtype=npu.int32)
    first = kernel[(1, )](out, 7, compile_mode=mode)
    npu.npu.synchronize()
    assert out.item() == 7
    second = kernel[(1, )](out, 9, compile_mode=mode)
    npu.npu.synchronize()
    assert out.item() == 9
    assert first.hash == second.hash
    assert second.src.signature["N"] == "i32"
    assert (1, ) not in second.src.constants


def test_ready_hit_does_not_compile_load_or_hash_toolchain(npu, monkeypatch):
    from triton.backends.ascend import compiler
    from triton.compiler.compiler import CompiledKernel
    torch = npu
    a, b = torch.ones(5000, device="npu"), torch.ones(5000, device="npu")
    out = torch.empty_like(a)
    first = pointwise[(3, )](a, b, out, 0.5, 5000, 5, 1024)

    def forbidden(*args, **kwargs):
        raise AssertionError("READY path prepared or read toolchain identity")

    monkeypatch.setattr(pointwise, "compile", forbidden)
    monkeypatch.setattr(compiler, "get_cann_version_file_hash", forbidden)
    initialize = CompiledKernel._init_handles

    def ready_only(kernel):
        if not kernel._initialization_complete:
            forbidden()
        return initialize(kernel)

    monkeypatch.setattr(CompiledKernel, "_init_handles", ready_only)
    selected = pointwise[(3, )](a, b, out, 0.5, 5000, 5, 1024)
    # Also cover a new desired tile: no foreground compiler work on a family hit.
    compatible = pointwise[(3, )](a, b, out, 0.5, 5000, 2, 4096)
    torch.npu.synchronize()
    assert selected.hash == compatible.hash == first.hash


def test_native_inference_recompiles_across_integer_widths(npu):
    kernel = triton.jit(compile_failure_probe.fn)
    out = npu.empty(1, device="npu", dtype=npu.int64)
    selected = []
    for value in (3, 4, 2**40, 2**41, 1):
        result = kernel[(1, )](out, value)
        npu.npu.synchronize()
        assert out.item() == value
        assert (1, ) not in result.src.constants
        assert result.src.fn.signature.parameters["N"].annotation is inspect.Parameter.empty
        selected.append(result)
    assert [kernel.src.signature["N"] for kernel in selected] == ["i32", "i32", "i64", "i64", "i32"]
    assert selected[0].hash == selected[1].hash == selected[4].hash
    assert selected[2].hash == selected[3].hash
    assert selected[0].hash != selected[2].hash


def test_large_integer_arithmetic_uses_native_width(npu):
    out = npu.empty(1, device="npu", dtype=npu.int64)
    selected = []
    for value in (3, 2**40, 2**41):
        kernel = arithmetic_width_probe[(1, )](out, value)
        npu.npu.synchronize()
        assert out.item() == value + 1
        assert (1, ) not in kernel.src.constants
        selected.append(kernel)
    assert [kernel.src.signature["N"] for kernel in selected] == ["i32", "i64", "i64"]
    assert selected[0].hash != selected[1].hash
    assert selected[1].hash == selected[2].hash


def test_float_data_dynamicizes_without_a_dtype_annotation(npu):
    x = npu.arange(8, device="npu", dtype=npu.float32)
    out = npu.empty_like(x)
    first = floating_scale_probe[(8, )](x, out, 1.25)
    second = floating_scale_probe[(8, )](x, out, 2.5)
    npu.npu.synchronize()
    assert npu.equal(out, x * 2.5)
    assert first.hash == second.hash
    assert second.src.signature["SCALE"] == "fp32"
    assert second.src.fn.signature.parameters["SCALE"].annotation is inspect.Parameter.empty
    assert (2, ) not in second.src.constants


def test_boolean_data_dynamicizes_while_control_flags_stay_static(npu):
    out = npu.empty(1, device="npu", dtype=npu.int32)
    first = compile_failure_probe[(1, )](out, True)
    second = compile_failure_probe[(1, )](out, False)
    npu.npu.synchronize()
    assert out.item() == 0
    assert first.hash == second.hash
    assert second.src.signature["N"] == "u1"
    assert (1, ) not in second.src.constants


def test_partial_coverage_and_offset_overlap_decline(npu):
    torch = npu
    a, b = torch.ones(5000, device="npu"), torch.ones(5000, device="npu")
    out = torch.full_like(a, -9)
    kernel = pointwise[(3, )](a, b, out, 0.5, 5000, 1, 2048)
    torch.npu.synchronize()
    assert kernel.src.constants[(6, )] == 2048
    assert torch.equal(out[:2048], torch.full_like(out[:2048], 1.5))
    assert torch.equal(out[2048:], torch.full_like(out[2048:], -9))


def test_worker_rebuilds_native_key_without_loading(npu, tmp_path):
    from triton._C.libtriton import get_cache_invalidating_env_vars
    from triton.backends.ascend.reuse.identity import variant_key
    from triton.backends.ascend.reuse.model import BuildSpec
    from triton.backends.ascend.reuse.protocol import build_request, environment, read_manifest
    from triton.backends.ascend.reuse.service import CompileWorker
    from triton.compiler import ASTSource
    torch = npu
    a = torch.ones(5000, device="npu")
    args = (a, a, a, 0.5, 5000, 5, 1024)
    _, _, target, backend, bind = pointwise.device_caches[0]
    bound, specialization, raw = bind(*args)
    opts, signature, constants, attrs = pointwise._pack_args(backend, {}, bound, specialization, raw)
    source = ASTSource(pointwise, signature, constants, attrs)
    key = variant_key(source, backend, opts, get_cache_invalidating_env_vars())
    spec = BuildSpec(pointwise, source, "", key, (), opts, tuple(specialization))
    request = build_request(spec, environment(0, target), 0)
    worker = CompileWorker()
    try:
        manifest = worker.compile(request)
        group = read_manifest(manifest, request)
        assert any(name.endswith(".npubin") for name in group)
    finally:
        worker.close()


def test_native_loading_concurrency_gate(npu, record_property):
    from triton.backends.ascend.reuse.context import exact_request_scope
    torch = npu
    a = torch.ones(5000, device="npu")
    b, out, isolated = a.clone(), torch.empty_like(a), torch.empty_like(a)
    with exact_request_scope():
        ready = pointwise[(3, )](a, b, out, 0.5, 5000, 5, 1024)
        cold = pointwise.warmup(a, b, isolated, 0.5, 5000, 3, 2048, grid=(3, ))
    torch.npu.synchronize()
    assert not cold._initialization_complete
    launch_times = []
    barrier = threading.Barrier(2)

    def load():
        torch.npu.set_device(0)
        barrier.wait()
        cold._init_handles()
        assert cold._initialization_complete and cold.module and cold.function

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(load)
        barrier.wait()
        for _ in range(50):
            start = time.perf_counter_ns()
            ready[(3, 1, 1)](a, b, out, 0.5, 5000, 5, 1024)
            torch.npu.synchronize()
            launch_times.append((time.perf_counter_ns() - start) / 1e6)
        future.result(timeout=60)
    assert torch.equal(out, a * 0.5 + b)
    record_property("concurrent_launch_ms", sorted(launch_times))


def test_dynamic_init_failure_falls_back_once_with_original_identity(npu, monkeypatch):
    from test_analysis import scalar_store
    from triton.backends.ascend.reuse.transform import ReuseASTSource
    from triton.compiler.compiler import CompiledKernel
    from triton.runtime.errors import OutOfResources
    torch = npu
    out = torch.full((16, ), -9, dtype=torch.int32, device="npu")
    calls, pre_runs = [], []
    original = CompiledKernel._init_handles

    def fail_dynamic(kernel):
        if isinstance(kernel.src, ReuseASTSource):
            raise OutOfResources(2, 1, "injected resources")
        return original(kernel)

    monkeypatch.setattr(CompiledKernel, "_init_handles", fail_dynamic)
    monkeypatch.setattr(scalar_store, "pre_run_hooks", [lambda *a, **kw: pre_runs.append(1)])

    def grid(meta):
        calls.append(1)
        return (7, )

    selected = scalar_store[grid](out, 3, 7)
    torch.npu.synchronize()
    assert calls == pre_runs == [1]
    assert selected.src.fn is scalar_store
    assert selected.src.constants[(1, )] == 3
    assert torch.equal(out[:7], torch.full_like(out[:7], 7))
    assert torch.equal(out[7:], torch.full_like(out[7:], -9))


def test_launch_hook_error_is_never_replayed(npu):
    from test_analysis import scalar_store
    from triton import knobs
    from triton.runtime.errors import OutOfResources
    out = npu.zeros(16, device="npu", dtype=npu.int32)
    calls = []

    def failure(metadata):
        calls.append(1)
        raise OutOfResources(2, 1, "injected launch hook")

    knobs.runtime.launch_enter_hook.add(failure)
    try:
        with pytest.raises(OutOfResources):
            scalar_store[(7, )](out, 3, 8)
    finally:
        knobs.runtime.launch_enter_hook.remove(failure)
    assert calls == [1]


def test_warmup_never_resolves_grid_or_publishes_ready(npu):
    from triton.backends.ascend.reuse.bridge import counters
    a = npu.ones(5000, device="npu")
    before = counters().get("ready_registered", 0)

    def grid(meta):
        raise AssertionError("warmup evaluated the grid")

    compiled = pointwise.warmup(a, a, a, 0.5, 5000, 2, 4096, grid=grid)
    assert not compiled._initialization_complete
    assert counters().get("ready_registered", 0) == before


def test_dynamic_reuse_uses_current_buffers_on_multiple_streams(npu):
    from test_analysis import scalar_store
    torch = npu
    outputs = [torch.full((32, ), -9, device='npu', dtype=torch.int32) for _ in range(2)]
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    torch.npu.synchronize()
    selected = []
    for output, stream, n, pad in zip(outputs, streams, (3, 4), (19, -13)):
        with torch.npu.stream(stream):
            selected.append(scalar_store[(32, )](output, n, pad))
    for stream in streams:
        stream.synchronize()
    assert selected[0].hash == selected[1].hash
    for output, n, pad in zip(outputs, (3, 4), (19, -13)):
        expected = torch.full_like(output, -9)
        expected[:2 * n + 1] = pad
        assert torch.equal(output, expected)


def test_original_global_change_check_is_not_hidden_by_snapshot(npu, monkeypatch):
    torch = npu
    out = torch.zeros(16, device='npu', dtype=torch.int32)
    selected = global_kernel[(16, )](out, 3)
    assert selected.src.fn is not global_kernel
    assert selected.src.signature['N'] == 'i32'
    torch.npu.synchronize()
    monkeypatch.setitem(global_kernel.__globals__, 'GLOBAL_BIAS', tl.constexpr(4))
    with pytest.raises(RuntimeError, match='Global variable GLOBAL_BIAS has changed'):
        global_kernel[(16, )](out, 4)


def test_fp32_tile_reuse_is_bitwise_for_non_dyadic_and_special_values(npu):
    from triton.backends.ascend.reuse.context import exact_request_scope
    torch = npu
    torch.manual_seed(314159)
    n = 65539
    a_cpu = torch.randn(n, dtype=torch.float32) * 1000
    b_cpu = torch.randn(n, dtype=torch.float32) / 7
    a_cpu[:8] = torch.tensor([0., -0., float('inf'), -float('inf'), float('nan'), 1e-30, -1e30, 1.])
    a, b = a_cpu.npu(), b_cpu.npu()
    reused, reference = torch.empty_like(a), torch.empty_like(a)
    first = pointwise[(7, )](a, b, reused, .123456789, n, (n + 1023) // 1024, 1024)
    selected = pointwise[(7, )](a, b, reused, .123456789, n, (n + 2047) // 2048, 2048)
    with exact_request_scope():
        exact = pointwise[(7, )](a, b, reference, .123456789, n, (n + 2047) // 2048, 2048)
    torch.npu.synchronize()
    assert first.hash == selected.hash and exact.src.constants[(6, )] == 2048
    assert torch.equal(reused.cpu().view(torch.int32), reference.cpu().view(torch.int32))


def test_identified_mlir_failure_falls_back_before_launch(npu, monkeypatch):
    from triton.backends.ascend.reuse.transform import FrozenJITFunction
    from triton.compiler.errors import MLIRCompilationError
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise MLIRCompilationError('test-stage', 'injected unsupported rewrite')

    monkeypatch.setattr(FrozenJITFunction, '_do_compile', fail)
    out = npu.zeros(1, device='npu', dtype=npu.int32)
    selected = compile_failure_probe[(1, )](out, 7)
    npu.npu.synchronize()
    assert calls == [1]
    assert selected.src.fn is compile_failure_probe and out.item() == 7


@pytest.mark.parametrize("old_tile,new_tile", [(1024, 1536), (1536, 1024), (1536, 768)])
@pytest.mark.parametrize("grid_type", [tuple, list])
def test_non_power_of_two_tiles_reuse_complete_binary(npu, old_tile, new_tile, grid_type):
    # A distinct source identity keeps each case out of previously registered families.
    kernel = triton.jit(pointwise.fn)
    result_name = f"result_{old_tile}_{new_tile}_{grid_type.__name__}"
    kernel._unsafe_update_src(pointwise.src.replace("result", result_name))
    a = npu.ones(5000, device='npu')
    out = npu.empty_like(a)
    grid = grid_type((3, ))
    first = kernel[grid](a, a, out, .5, 5000, triton.cdiv(5000, old_tile), old_tile)
    npu.npu.synchronize()
    out.fill_(-1)
    selected = kernel[grid](a, a, out, .5, 5000, triton.cdiv(5000, new_tile), new_tile)
    npu.npu.synchronize()
    assert selected.hash == first.hash
    assert selected.src.constants[(6, )] == old_tile
    assert npu.equal(out, npu.full_like(out, 1.5))


def test_retained_constants_keep_boolean_integer_and_float_bits_distinct(npu):
    out = npu.zeros(1, device='npu', dtype=npu.int32)
    hashes = []
    for n, marker in ((3, 0.0), (4, -0.0), (5, True), (6, 1)):
        selected = constant_identity_probe[(1, )](out, n, marker)
        npu.npu.synchronize()
        assert selected.src.signature['N'] == 'i32'
        assert selected.src.signature['MARKER'] == 'constexpr'
        assert out.item() == n
        hashes.append(selected.hash)
    assert len(set(hashes)) == 4


@pytest.mark.parametrize("name,n,expected,dynamic", [
    ("same_line_helpers", 3, 9, True),
    ("different_helper_parameters", 3, 6, True),
    ("nested_helper_store", 3, 6, True),
    ("inverted_division", 4, -2, True),
    ("inverted_division", -5, 2, True),
    ("inverted_remainder", 4, -1, True),
    ("float_default_entry", 16777219, 1677722, True),
])
def test_scalar_reuse_review_regressions(npu, name, n, expected, dynamic):
    import test_regressions as kernels
    fn = getattr(kernels, name)
    out = npu.zeros(1, device="npu", dtype=npu.int32)
    selected = fn[(1, )](out, n)
    npu.npu.synchronize()
    assert out.item() == expected
    assert selected.src.signature["N"] == ("i32" if dynamic else "constexpr")


def test_explicit_unsigned_scalar_uses_native_dynamic_promotion(npu):
    from test_regressions import scalar_annotation_kernel
    fn = scalar_annotation_kernel("u16")
    out = npu.zeros(1, device="npu", dtype=npu.int32)
    selected = fn[(1, )](out, 5, 3)
    npu.npu.synchronize()
    assert out.item() == -9
    assert selected.src.fn is not fn
    assert selected.src.signature["x"] == "u16"
    assert selected.src.signature["N"] == "i32"
    assert (2, ) not in selected.src.constants


@pytest.mark.parametrize("case", ["bound_value", "loop_carried"])
def test_tile_reuse_rejects_iteration_dependent_results(npu, case):
    from test_regressions import pointwise_copy
    replacement = "result = x + work" if case == "bound_value" else "result = x * scale\n        scale = scale + 1"
    fn = pointwise_copy("result = x * scale", replacement)
    a = npu.ones(1024, device="npu", dtype=npu.float32)
    out = npu.empty_like(a)
    fn[(1, )](a, out, 1024, 4, 256, 0.5)
    selected = fn[(1, )](a, out, 1024, 2, 512, 0.5)
    npu.npu.synchronize()
    assert selected.src.constants[(4, )] == 512
    expected = npu.full_like(a, 3.0 if case == "bound_value" else 0.5)
    if case == "loop_carried":
        expected[512:] = 1.5
    assert npu.equal(out, expected)
