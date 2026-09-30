"""Reuse checks against the actual autotune layernorm example."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import triton.language as tl
from test_tile_rules import tensors as tensor_fixture
from triton.backends.ascend.reuse.analysis import analyze
from triton.backends.ascend.reuse.context import InvocationFrame
from triton.backends.ascend.reuse.dispatch import resolve_dispatch
from triton.backends.ascend.reuse.guards import adapt_tile
from triton.backends.ascend.reuse.identity import family_key
from triton.backends.ascend.reuse.local_axis import match_local_axis
from triton.backends.ascend.reuse.model import ParameterClass, PreparedLaunch

tensors = tensor_fixture


def reference_kernel(replace=None):
    path = Path(__file__).resolve().parents[1] / "autotune_ut/03-layer-norm.py"
    source = path.read_text()
    if replace:
        assert replace[0] in source
        source = source.replace(*replace)
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_layer_norm_fwd_fused")
    return SimpleNamespace(
        arg_names=tuple(arg.arg for arg in function.args.args), params=tuple(
            SimpleNamespace(is_constexpr=i in (10,
                                               11), name=arg.arg, num=i, annotation_type="", do_not_specialize=False)
            for i, arg in enumerate(function.args.args)), parse=lambda: ast.parse(ast.unparse(function)),
        get_capture_scope=lambda: {"tl": tl})


def test_rblock_changes_without_changing_xblock_or_family():
    recipe = match_local_axis(reference_kernel())
    assert recipe is not None and recipe.tile == 11
    profile = SimpleNamespace(recipe=recipe, source_key="layernorm", plan=SimpleNamespace(dynamic=()))
    options = SimpleNamespace(hash=lambda: "options")
    target = SimpleNamespace(backend="npu", arch="any")

    def family(xblock, rblock):
        source = SimpleNamespace(constants={(10, ): xblock, (11, ): rblock}, signature={}, attrs={})
        return family_key(profile, source, options, target, {})

    assert family(2, 32) == family(2, 64)
    assert family(2, 32) != family(4, 32)


def test_layernorm_rule_is_independent_of_local_names():
    fn = reference_kernel(("RBLOCK_SIZE", "REDUCTION_TILE"))
    assert match_local_axis(fn) is not None


def test_layernorm_parameter_positions_are_not_a_template():
    fn = reference_kernel()
    tree = fn.parse()
    args = tree.body[0].args.args
    args.insert(0, args.pop(11))
    fn.arg_names = tuple(arg.arg for arg in args)
    fn.params = tuple(SimpleNamespace(is_constexpr=i in (0, 11)) for i in range(12))
    fn.parse = lambda: tree
    recipe = match_local_axis(fn)
    assert recipe is not None and recipe.tile == 0


@pytest.mark.parametrize("old,new", [
    ("tl.program_id(0) * XBLOCK_SIZE", "tl.program_id(0) * RBLOCK_SIZE"),
    ("range(0, N, RBLOCK_SIZE)", "range(0, N, XBLOCK_SIZE)"),
    ("col_idx < N", "col_idx <= N"),
    ("x = tl.where(mask, x - mean, 0.0)", "x = x - mean"),
    ("tl.sum(_mean, axis=1, keep_dims=True)", "tl.max(_mean, axis=1, keep_dims=True)"),
    ("_mean += a", "_mean += a + RBLOCK_SIZE"),
])
def test_changed_reduction_or_partition_is_not_a_proof(old, new):
    assert match_local_axis(reference_kernel((old, new))) is None


def layernorm_arguments(tensors, rows=5, n=67, rblock=32):
    sizes = (rows * n, rows * n, n, n, rows, rows)
    pointers = tuple(tensors(65536 * (i + 1), width=2 if i < 4 else 4, count=size) for i, size in enumerate(sizes))
    return (*pointers, n, n, rows, 1e-5, 2, rblock)


def test_layernorm_guard_preserves_outer_arguments(tensors):
    recipe = match_local_axis(reference_kernel())
    args = layernorm_arguments(tensors)
    adapted = adapt_tile(recipe, args, {(10, ): 2, (11, ): 16}, (3, 1, 1), 0, None, compile_mode="simd")
    assert adapted is not None and adapted[:-1] == args[:-1] and adapted[-1] == 16


def test_actual_analysis_classifies_only_rblock_as_reusable(tensors):
    fn = reference_kernel()
    profile = analyze(fn, dict(zip(fn.arg_names, layernorm_arguments(tensors))), "layernorm")
    decisions = {item.position: item.classification for item in profile.decisions}
    assert decisions[11] is ParameterClass.ScheduleReusable
    assert decisions[10] is not ParameterClass.ScheduleReusable
    assert profile.recipe is not None and profile.recipe.local_loop


def test_callable_grid_is_resolved_once_with_requested_outer_partition(tensors):
    fn = reference_kernel()
    arguments = layernorm_arguments(tensors)
    recipe = match_local_axis(fn)
    types = ["*fp16"] * 6 + ["i32", "i32", "i32", "fp32", "constexpr", "constexpr"]
    signature = dict(zip(fn.arg_names, types))

    def binder(*args, **kwargs):
        return {}, [(ty, value if ty == "constexpr" else None) for ty, value in zip(types, args)], {}

    candidates = [
        SimpleNamespace(variant_key=str(tile), source=SimpleNamespace(signature=signature, constants={(10, ): 2, (11, ):
                                                                                                      tile}, attrs={}))
        for tile in (0, 16)
    ]
    registry = SimpleNamespace(lookup_exact_ready=lambda *args: None, lookup_family_ready=lambda *args: candidates,
                               has_registration=lambda *args: True, event=lambda *args: None)
    owner = SimpleNamespace(arg_names=fn.arg_names, device_caches={0: (None, None, None, None, binder)})
    spec = SimpleNamespace(owner=owner, variant_key="32", family_key="family",
                           options=SimpleNamespace(compile_mode="simd"))
    frame, calls = InvocationFrame(), []

    def grid(meta):
        calls.append(meta["RBLOCK_SIZE"])
        return [(meta["M"] + meta["XBLOCK_SIZE"] - 1) // meta["XBLOCK_SIZE"], 1, 1]

    result = resolve_dispatch(registry, spec, SimpleNamespace(recipe=recipe), arguments, grid, (1, 0), None,
                              resolve_grid=lambda: frame.resolve_grid(grid, dict(zip(fn.arg_names, arguments))))
    assert isinstance(result, PreparedLaunch) and result.arguments[-1] == 16
    assert frame.resolve_grid(grid, dict(zip(fn.arg_names, result.arguments))) == [3, 1, 1]
    assert calls == [32]


@pytest.mark.parametrize("grid", [(3, 2)])
def test_layernorm_rejects_incomplete_or_duplicate_outer_coverage(tensors, grid):
    recipe = match_local_axis(reference_kernel())
    assert adapt_tile(recipe, layernorm_arguments(tensors), {(11, ): 16}, grid, 0, None, compile_mode="simd") is None


def test_same_coordinate_statistic_aliases_preserve_store_order(tensors):
    recipe = match_local_axis(reference_kernel())
    args = list(layernorm_arguments(tensors))
    args[4] = args[5]
    assert adapt_tile(recipe, args, {(11, ): 16}, (3, ), 0, None, compile_mode="simd") is not None


def execute_reference(x, w, bias, row_tile, reduction_tile):
    """Execute the example's actual Python AST with NumPy-backed TL operations."""

    class Tile(np.ndarray):

        def to(self, dtype):
            return self.astype(dtype)

    class Pointer:

        def __init__(self, data, offsets=0):
            self.data = data
            self.offsets = offsets

        def __add__(self, offsets):
            return Pointer(self.data, self.offsets + offsets)

    def load(pointer, mask, other=0):
        offsets, mask = np.broadcast_arrays(pointer.offsets, mask)
        result = np.full(offsets.shape, other, dtype=pointer.data.dtype)
        result[mask] = pointer.data.reshape(-1)[offsets[mask]]
        return result.view(Tile)

    def store(pointer, value, mask):
        offsets, value, mask = np.broadcast_arrays(pointer.offsets, value, mask)
        pointer.data.reshape(-1)[offsets[mask]] = value[mask]

    functions = SimpleNamespace(arange=lambda start, end: np.arange(start, end, dtype=np.int32).view(Tile),
                                zeros=lambda shape, dtype: np.zeros(shape, dtype=dtype).view(Tile), float32=np.float32,
                                load=load, store=store,
                                sum=lambda value, axis, keep_dims: value.sum(axis=axis, keepdims=keep_dims),
                                where=lambda condition, a, b: np.where(condition, a, b).view(Tile), sqrt=np.sqrt)
    function = reference_kernel().parse().body[0]
    function.decorator_list = []
    for argument in function.args.args:
        argument.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    scope = {"tl": functions}
    exec(compile(module, "layernorm_host.py", "exec"), scope)  # noqa: S102 - execute the repository fixture only
    output = np.full_like(x, np.nan)
    mean = np.full(x.shape[0], np.nan, dtype=np.float32)
    rstd = np.full_like(mean, np.nan)
    for program in range((x.shape[0] + row_tile - 1) // row_tile):
        functions.program_id = lambda axis, program=program: program
        scope[function.name](Pointer(x), Pointer(output), Pointer(w), Pointer(bias), Pointer(mean), Pointer(rstd),
                             x.shape[1], x.shape[1], x.shape[0], 1e-5, row_tile, reduction_tile)
    return output, mean, rstd


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("rblock", [16, 32, 48, 128])
def test_reference_reduction_tiles_preserve_layernorm_with_tolerance(dtype, rblock):
    rng = np.random.default_rng(5)
    x = rng.normal(-2.3, 0.5, (5, 67)).astype(dtype)
    w = rng.random(67).astype(dtype)
    bias = rng.random(67).astype(dtype)
    output, mean, rstd = execute_reference(x, w, bias, 2, rblock)
    values = x.astype(np.float32)
    expected_mean = values.mean(axis=1)
    expected_rstd = 1 / np.sqrt(((values - expected_mean[:, None])**2).mean(axis=1) + 1e-5)
    expected = ((values - expected_mean[:, None]) * expected_rstd[:, None] * w + bias).astype(dtype)
    np.testing.assert_allclose(output, expected, atol=1e-2, rtol=0)
    np.testing.assert_allclose(mean, expected_mean, atol=1e-5, rtol=0)
    np.testing.assert_allclose(rstd, expected_rstd, atol=1e-5, rtol=0)
