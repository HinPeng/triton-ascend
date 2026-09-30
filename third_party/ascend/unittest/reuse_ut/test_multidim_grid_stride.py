"""Coverage and dispatch proofs for flattened multi-dimensional grid-stride loops."""
import itertools
from types import SimpleNamespace

import pytest

from test_grid_stride_callable import dispatch_objects
from test_tile_rules import kernel
from test_tile_rules import tensors as tensor_fixture
from triton.backends.ascend.reuse.analysis import analyze
from triton.backends.ascend.reuse.context import InvocationFrame
from triton.backends.ascend.reuse.dispatch import resolve_dispatch
from triton.backends.ascend.reuse.guards import TILE_INDEX_MAX, adapt_tile
from triton.backends.ascend.reuse.model import JitTailRequest, ParameterClass, PreparedLaunch
from triton.backends.ascend.reuse.rules import match_grid_stride

tensors = tensor_fixture


def flatten(axes, form="nested"):
    pid = lambda axis: f"tl.program_id({axis})"
    size = lambda axis: f"tl.num_programs({axis})"
    if form == "nested":
        start = pid(axes[-1])
        for axis in reversed(axes[:-1]):
            start = f"{pid(axis)} + {size(axis)} * ({start})"
    else:
        terms = [" * ".join([pid(axis), *(size(a) for a in axes[:i])]) for i, axis in enumerate(axes)]
        start = " + ".join(reversed(terms))
    return start, " * ".join(size(axis) for axis in reversed(axes))


def grid_kernel(start, step, result="x + length"):
    return kernel(f"range({start}, work, {step})", "block * tile + tl.arange(0, tile)", result=result)


def arguments(tensors, n=5000, tile=32):
    return (tensors(65536, count=n), tensors(131072, count=n), n, (n + tile - 1) // tile, tile)


GRID_ORDERS = [(grid, order)
               for grid in ((2, 3), (2, 3, 4), (1, 3, 4), (2, 1, 4), (2, 3, 1))
               for order in itertools.permutations(range(len(grid)))]


@pytest.mark.parametrize("grid,order", GRID_ORDERS)
@pytest.mark.parametrize("form", ["nested", "expanded"])
@pytest.mark.parametrize("old,requested", [(16, 32), (32, 16)])
@pytest.mark.parametrize("n", [67, 5000])
def test_multidimensional_flattening_covers_each_element_once(tensors, grid, order, form, old, requested, n):
    start, step = flatten(order, form)
    recipe = match_grid_stride(grid_kernel(start, step))
    assert recipe is not None and not recipe.local_loop
    args = arguments(tensors, n, requested)
    adapted = adapt_tile(recipe, args, {(4, ): old}, grid, 0, compile_mode="simd")
    assert adapted is not None and adapted[3:] == ((n + old - 1) // old, old)
    # Execute the source expressions with independent Python program IDs/lanes.
    dimensions = (*grid, *((1, ) * (3 - len(grid))))
    visited, starts = [], []
    for ids in itertools.product(*(range(size) for size in dimensions)):
        scope = {"tl": SimpleNamespace(program_id=lambda axis: ids[axis], num_programs=lambda axis: dimensions[axis])}
        first = eval(start, {"__builtins__": {}}, scope)
        stride = eval(step, {"__builtins__": {}}, scope)
        starts.append(first)
        for block in range(first, adapted[3], stride):
            visited.extend(block * old + lane for lane in range(old) if block * old + lane < n)
    assert sorted(starts) == list(range(len(starts)))
    assert sorted(visited) == list(range(n))


@pytest.mark.parametrize("grid,axes", [((1, 7), (1, )), ((1, 1, 7), (2, )), ((3, 1, 2), (2, 0))])
def test_unused_axes_must_be_singletons(tensors, grid, axes):
    recipe = match_grid_stride(grid_kernel(*flatten(axes)))
    assert recipe is not None
    args = arguments(tensors)
    assert adapt_tile(recipe, args, {(4, ): 16}, grid, 0, compile_mode="simd_simt_template") is not None
    omitted = next(axis for axis in range(len(grid)) if axis not in axes)
    overlapping = list(grid)
    overlapping[omitted] = 2
    assert adapt_tile(recipe, args, {(4, ): 16}, tuple(overlapping), 0, compile_mode="simd") is None


@pytest.mark.parametrize("start,step", [
    ("tl.program_id(0)", "tl.num_programs(0)"),
    ("tl.program_id(0) + tl.program_id(1)", "tl.num_programs(0) * tl.num_programs(1)"),
    ("tl.program_id(0) * 2 + tl.program_id(1) * tl.num_programs(0) * 2", "tl.num_programs(0) * tl.num_programs(1)"),
    ("1 + tl.program_id(0) + tl.num_programs(0) * tl.program_id(1)", "tl.num_programs(0) * tl.num_programs(1)"),
    ("tl.program_id(0) + tl.num_programs(0) * tl.program_id(1)", "tl.num_programs(0)"),
    ("tl.program_id(0) + tl.num_programs(0) * tl.program_id(1)", "0"),
    ("tl.program_id(0) + tl.num_programs(0) * tl.program_id(1)", "0 - tl.num_programs(0) * tl.num_programs(1)"),
])
def test_overlapping_incomplete_or_shifted_mapping_is_rejected(tensors, start, step):
    recipe = match_grid_stride(grid_kernel(start, step))
    assert recipe is not None
    assert adapt_tile(recipe, arguments(tensors), {(4, ): 16}, (2, 3), 0, compile_mode="simd") is None


@pytest.mark.parametrize("start,step", [
    ("tl.program_id(0) * tl.program_id(1)", "tl.num_programs(0) * tl.num_programs(1)"),
    ("tl.program_id(0) + tile - tile", "tl.num_programs(0)"),
    ("tl.program_id(0) + work - work", "tl.num_programs(0)"),
    ("tl.program_id(0) + 0.0", "tl.num_programs(0)"),
    ("tl.program_id(0) + False", "tl.num_programs(0)"),
    ("tl.program_id(0)", "tl.num_programs(0) + tl.program_id(1)"),
    ("tl.program_id(0)", "tl.num_programs(0) + length - length"),
])
def test_noninteger_nonlinear_or_argument_dependent_schedule_is_not_proved(start, step):
    assert match_grid_stride(grid_kernel(start, step)) is None


@pytest.mark.parametrize("result",
                         ["x + tl.program_id(1)", "x + tl.num_programs(2)", "x + tile", "x + block", "x + work"])
def test_multidimensional_schedule_cannot_leak_into_values(result):
    assert match_grid_stride(grid_kernel(*flatten((0, 1, 2)), result=result)) is None


@pytest.mark.parametrize("field", ["start", "step"])
def test_canceled_grid_intermediate_overflow_is_rejected(tensors, field):
    start, step = flatten((0, 1))
    overflow = "tl.num_programs(0) * tl.num_programs(0)"
    if field == "start":
        start = f"({start}) + ({overflow} - {overflow})"
    else:
        step = f"({step}) + ({overflow} - {overflow})"
    recipe = match_grid_stride(grid_kernel(start, step))
    assert recipe is not None
    assert adapt_tile(recipe, arguments(tensors), {(4, ): 16}, (50000, 1), 0, compile_mode="simd") is None


def test_grid_product_overflow_is_rejected(tensors):
    recipe = match_grid_stride(grid_kernel(*flatten((0, 1))))
    assert adapt_tile(recipe, arguments(tensors), {(4, ): 16}, (65536, 65536), 0, compile_mode="simd") is None


@pytest.mark.parametrize("old,requested", [(16, 32), (32, 16)])
def test_loop_increment_bounds_check_both_tile_configurations(tensors, old, requested):
    recipe = match_grid_stride(grid_kernel(*flatten((0, 1))))
    args = arguments(tensors, 67, requested)
    assert adapt_tile(recipe, args, {(4, ): old}, (TILE_INDEX_MAX - 4, 1), 0, compile_mode="simd") is not None
    assert adapt_tile(recipe, args, {(4, ): old}, (TILE_INDEX_MAX - 3, 1), 0, compile_mode="simd") is None


@pytest.mark.parametrize("grid", [(2, 3), (2, 3, 4)])
@pytest.mark.parametrize("grid_kind", ["tuple", "list", "callable"])
@pytest.mark.parametrize("old,requested", [(1024, 2048), (2048, 1024)])
def test_multidimensional_dispatch_preserves_grid(tensors, grid, grid_kind, old, requested):
    fn, args, profile, registry, spec, _ = dispatch_objects(tensors, requested, (0, old))
    profile.recipe = match_grid_stride(grid_kernel(*flatten(tuple(range(len(grid))))))
    frame, calls = InvocationFrame(), []

    def callback(meta):
        calls.append((meta["tile"], meta["work"]))
        return [meta["work"], *grid[1:]]

    launch_grid = callback if grid_kind == "callable" else (list(grid) if grid_kind == "list" else grid)
    result = resolve_dispatch(registry, spec, profile, args, launch_grid, (1, 0), None,
                              resolve_grid=lambda: frame.resolve_grid(launch_grid, dict(zip(fn.arg_names, args))))
    assert isinstance(result, PreparedLaunch)
    assert result.grid is None and result.arguments[3:] == ((5000 + old - 1) // old, old)
    actual_grid = frame.resolve_grid(launch_grid, dict(zip(fn.arg_names, result.arguments)))
    expected = [args[3], *grid[1:]] if grid_kind == "callable" else launch_grid
    assert actual_grid == expected
    assert calls == ([(requested, args[3])] if grid_kind == "callable" else [])


def test_bad_multidimensional_candidate_preserves_callable_for_fallback(tensors):
    fn, args, profile, registry, spec, _ = dispatch_objects(tensors)
    start, _ = flatten((0, 1))
    profile.recipe = match_grid_stride(grid_kernel(start, "tl.num_programs(0)"))
    frame, calls = InvocationFrame(), []

    def callback(meta):
        calls.append(meta["tile"])
        return (2, 3)

    result = resolve_dispatch(registry, spec, profile, args, callback, (1, 0), None,
                              resolve_grid=lambda: frame.resolve_grid(callback, dict(zip(fn.arg_names, args))))
    assert isinstance(result, JitTailRequest)
    assert frame.resolve_grid(callback, dict(zip(fn.arg_names, args))) == (2, 3)
    assert calls == [2048] and args[3:] == (3, 2048)


def test_analysis_selects_multidimensional_grid_stride_recipe(tensors):
    fn = grid_kernel(*flatten((2, 0, 1)))
    fn.params = tuple(
        SimpleNamespace(name=name, num=i, is_constexpr=i == 4, annotation_type="", do_not_specialize=False)
        for i, name in enumerate(fn.arg_names))
    args = arguments(tensors)
    profile = analyze(fn, dict(zip(fn.arg_names, args)), "multidimensional-grid-stride")
    assert profile.recipe is not None and not profile.recipe.local_loop
    assert next(item for item in profile.decisions
                if item.position == 4).classification is ParameterClass.ScheduleReusable
    assert adapt_tile(profile.recipe, args, {(4, ): 16}, (2, 3, 4), 0, compile_mode="simd") is not None
