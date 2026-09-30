"""Direct pid*tile+lane coverage, dispatch grid changes and rejection guards."""
import inspect
import itertools
import sys
from types import ModuleType, SimpleNamespace

import pytest
from test_axis_generalization import helper, with_helpers
from test_local_axis import source_kernel
from test_tile_rules import tensors as tensor_fixture
from triton.backends.ascend.reuse.analysis import analyze
from triton.backends.ascend.reuse.context import InvocationFrame
from triton.backends.ascend.reuse.dispatch import resolve_dispatch
from triton.backends.ascend.reuse.footprint import adapt_grid
from triton.backends.ascend.reuse.guards import adapt_tile
from triton.backends.ascend.reuse.local_axis import match_local_axis
from triton.backends.ascend.reuse.model import JitTailRequest, ParameterClass, PreparedLaunch

tensors = tensor_fixture

DIRECT = """
index = tl.program_id(0) * width + tl.arange(0, width)
valid = index < length
x = tl.load(src + index, mask=valid, other=0.0)
tl.store(dst + index, x * 2.0 + index, mask=valid)
"""


@pytest.mark.parametrize("n", [1, 63, 64, 67, 5000])
@pytest.mark.parametrize("old,new", [(16, 32), (32, 16), (1536, 1024), (1024, 1536)])
@pytest.mark.parametrize("mode", ["simd", "simd_simt_template"])
def test_direct_grid_tile_visits_each_valid_element_once(tensors, n, old, new, mode):
    recipe = match_local_axis(source_kernel(DIRECT))
    assert recipe is not None and recipe.grid_axis == 0 and not recipe.local_loop
    arguments = (tensors(65536, count=n), tensors(131072, count=n), n, new)
    requested_grid = ((n + new - 1) // new, )
    constants = {(3, ): old}
    actual = adapt_tile(recipe, arguments, constants, requested_grid, 0, None, compile_mode=mode)
    selected_grid = adapt_grid(recipe, arguments, constants, requested_grid)
    assert actual is not None and actual[3] == old
    assert selected_grid == ((n + old - 1) // old, )
    addresses = [
        pid * actual[3] + lane
        for pid in range(selected_grid[0])
        for lane in range(actual[3])
        if pid * actual[3] + lane < n
    ]
    assert addresses == list(range(n))


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_other_grid_axes_keep_their_partition(tensors, axis):
    others = [i for i in range(3) if i != axis]
    row = f"(tl.program_id({others[1]}) * tl.num_programs({others[0]}) + tl.program_id({others[0]}))"
    body = DIRECT.replace("tl.program_id(0)", f"tl.program_id({axis})")
    body = f"base = {row} * length\n" + body.replace("src + index", "src + base + index").replace(
        "dst + index", "dst + base + index")
    recipe = match_local_axis(source_kernel(body))
    assert recipe is not None and recipe.grid_axis == axis
    arguments = (tensors(65536, count=402), tensors(131072, count=402), 67, 32)
    grid = [1, 1, 1]
    grid[axis], grid[others[0]], grid[others[1]] = 3, 2, 3
    selected = adapt_grid(recipe, arguments, {(3, ): 16}, tuple(grid))
    actual = adapt_tile(recipe, arguments, {(3, ): 16}, tuple(grid), 0, None, compile_mode="simd")
    assert actual is not None and selected[axis] == 5
    assert all(selected[i] == grid[i] for i in others)
    visited = []
    for ids in itertools.product(*(range(size) for size in selected)):
        base = (ids[others[1]] * grid[others[0]] + ids[others[0]]) * 67
        visited.extend(base + ids[axis] * 16 + lane for lane in range(16) if ids[axis] * 16 + lane < 67)
    assert sorted(visited) == list(range(402))


def test_direct_tile_classification(tensors):
    fn = source_kernel(DIRECT)
    for i, (name, parameter) in enumerate(zip(fn.arg_names, fn.params)):
        parameter.num, parameter.name = i, name
        parameter.annotation_type, parameter.do_not_specialize = "", False
    args = (tensors(65536), tensors(131072), 67, 32)
    profile = analyze(fn, dict(zip(fn.arg_names, args)), "direct-tile")
    assert profile.recipe.grid_axis == 0
    assert next(d for d in profile.decisions if d.position == 3).classification is ParameterClass.ScheduleReusable


def test_grid_coordinate_can_be_built_in_a_helper(tensors):
    make = helper("""
def indices(size):
    return tl.arange(0, size) + size * tl.program_id(0)
""", constexpr=("size", ))
    body = DIRECT.replace("tl.program_id(0) * width + tl.arange(0, width)", "indices(size=width)")
    recipe = match_local_axis(with_helpers(source_kernel(body), indices=make))
    assert recipe is not None and recipe.grid_axis == 0
    args = (tensors(65536), tensors(131072), 67, 32)
    assert adapt_tile(recipe, args, {(3, ): 16}, (3, ), 0, None, compile_mode="simd") is not None


@pytest.mark.parametrize("replacement", [
    "x + tl.program_id(0)",
    "x + tl.num_programs(0)",
    "x + tl.exp(tl.program_id(0))",
    "tl.where(valid, tl.program_id(0), x)",
    "x + width",
])
def test_partition_dependent_values_are_not_reused(replacement):
    assert match_local_axis(source_kernel(DIRECT.replace("x * 2.0 + index", replacement))) is None


@pytest.mark.parametrize("body", [
    DIRECT.replace("src + index", "src + index + tl.program_id(0)"),
    DIRECT.replace("index < length", "index < tl.num_programs(0)"),
    DIRECT.replace("index < length", "index < width"),
    DIRECT.replace("mask=valid", "mask=True"),
    DIRECT.replace("x * 2.0 + index", "tl.sum(x, axis=0)"),
    DIRECT + "tl.store(dst, 1.0, mask=length > 0)\n",
])
def test_missing_coverage_or_per_program_effects_decline(body):
    assert match_local_axis(source_kernel(body)) is None


@pytest.mark.parametrize("grid", [(1, ), (2, ), (4, ), (3, 2)])
def test_incomplete_or_duplicate_requested_grid_declines(tensors, grid):
    recipe = match_local_axis(source_kernel(DIRECT))
    args = (tensors(65536), tensors(131072), 67, 32)
    assert adapt_tile(recipe, args, {(3, ): 16}, grid, 0, None, compile_mode="simd") is None


def dispatch_objects(tensors, tiles=(0, 16), grid_num_tiles=None):
    fn = source_kernel(DIRECT)
    arguments = (tensors(65536), tensors(131072), 67, 32)
    recipe = match_local_axis(fn)
    types = ["*fp16", "*fp16", "i32", "constexpr"]
    signature = dict(zip(fn.arg_names, types))

    def binder(*args, **kwargs):
        return dict(zip(fn.arg_names,
                        args)), [(ty, value if ty == "constexpr" else None) for ty, value in zip(types, args)], kwargs

    candidates = [
        SimpleNamespace(variant_key=str(tile), kernel=object(),
                        source=SimpleNamespace(signature=signature, constants={(3, ): tile}, attrs={}))
        for tile in tiles
    ]
    registry = SimpleNamespace(lookup_exact_ready=lambda *args: None, lookup_family_ready=lambda *args: candidates,
                               has_registration=lambda *args: True, event=lambda *args: None,
                               disabled=lambda *args: False)
    owner = SimpleNamespace(arg_names=fn.arg_names, device_caches={0: (None, None, None, None, binder)})
    spec = SimpleNamespace(owner=owner, variant_key="requested", family_key="family",
                           options=SimpleNamespace(compile_mode="simd", grid_num_tiles=grid_num_tiles))
    return fn, arguments, recipe, registry, spec


@pytest.mark.parametrize("callable_grid", [False, True])
def test_dispatch_returns_grid_and_frame_launches_it_once(tensors, callable_grid):
    fn, args, recipe, registry, spec = dispatch_objects(tensors)
    frame, calls = InvocationFrame(), []

    def grid(meta):
        calls.append(meta["width"])
        return [(meta["length"] + meta["width"] - 1) // meta["width"]]

    requested = grid if callable_grid else (3, )
    result = resolve_dispatch(registry, spec, SimpleNamespace(recipe=recipe), args, requested, (1, 0), None,
                              resolve_grid=lambda: frame.resolve_grid(grid, dict(zip(fn.arg_names, args))))
    assert isinstance(result, PreparedLaunch) and result.grid == (5, ) and result.arguments[3] == 16
    frame.select_grid(result.grid)
    assert frame.resolve_grid(requested, dict(zip(fn.arg_names, result.arguments))) == (5, )
    assert calls == ([32] if callable_grid else [])


@pytest.mark.parametrize("grid_num_tiles,tiles", [(None, (0, )), (3, (16, ))])
def test_rejected_candidates_leave_requested_grid_for_fallback(tensors, grid_num_tiles, tiles):
    fn, args, recipe, registry, spec = dispatch_objects(tensors, tiles, grid_num_tiles)
    frame, calls = InvocationFrame(), []

    def grid(meta):
        calls.append(meta["width"])
        return (3, )

    result = resolve_dispatch(registry, spec, SimpleNamespace(recipe=recipe), args, grid, (1, 0), None,
                              resolve_grid=lambda: frame.resolve_grid(grid, dict(zip(fn.arg_names, args))))
    assert isinstance(result, JitTailRequest)
    assert frame.resolve_grid(grid, dict(zip(fn.arg_names, args))) == (3, ) and calls == [32]


def test_frame_cannot_change_grid_after_launch():
    frame = InvocationFrame()
    frame.before_launch()
    with pytest.raises(RuntimeError, match="after launch"):
        frame.select_grid((5, ))


@pytest.mark.parametrize("tiles,expected_grid,expected_tile", [((16, ), (5, ), 16), ((0, ), (3, ), 32)])
def test_bridge_passes_selected_or_original_grid_to_jit_tail(monkeypatch, tensors, tiles, expected_grid, expected_tile):
    from triton.backends.ascend.reuse import bridge, identity
    from triton.backends.ascend.reuse.dsl import source
    from triton.runtime import jit

    fn, arguments, _, registry, spec = dispatch_objects(tensors, tiles)
    options = spec.options
    options.ir_override = None
    options.hash = lambda: "options"
    target = SimpleNamespace(backend="npu", arch="any")
    backend = SimpleNamespace(target=target, parse_options=lambda kwargs: options)
    binder = spec.owner.device_caches[0][-1]
    fn.device_caches = {0: ({}, {}, target, backend, binder)}
    fn.signature = inspect.Signature(
        [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in fn.arg_names])
    for i, (name, parameter) in enumerate(zip(fn.arg_names, fn.params)):
        parameter.num, parameter.name = i, name
        parameter.annotation_type, parameter.do_not_specialize = "", False
    signature = dict(zip(fn.arg_names, ("*fp16", "*fp16", "i32", "constexpr")))
    fn._pack_args = lambda backend, kwargs, bound, specialization, raw: (options, signature, {(3, ): bound["width"]}, {
                                                                                              })
    fn.ASTSource = lambda owner, signature, constants, attrs: SimpleNamespace(fn=owner, signature=signature, constants=
                                                                              constants, attrs=attrs)

    def module(name, **attributes):
        value = ModuleType(name)
        value.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, value)

    module("triton._C.libtriton", get_cache_invalidating_env_vars=dict)
    module("triton.backends.ascend.compiler", NPUOptions=SimpleNamespace(__dataclass_fields__={}))
    module("triton.backends.ascend.reuse.transform", FrozenJITFunction=object, changed=lambda plan: False)
    module("triton.backends.ascend.reuse.service", PreparationService=object)
    module("triton.compiler.errors", CompilationError=type("CompilationError", (Exception, ), {}),
           MLIRCompilationError=type("MLIRCompilationError", (Exception, ), {}))
    module("triton.runtime.errors", OutOfResources=type("OutOfResources", (Exception, ), {}))
    monkeypatch.setattr(bridge, "requires_original", lambda *args: False)
    monkeypatch.setattr(jit, "compute_cache_key", lambda *args: "memory", raising=False)
    monkeypatch.setattr(source, "source_stamp", lambda fn: SimpleNamespace(token="source", matches=lambda: True))
    monkeypatch.setattr(identity, "variant_key", lambda *args: "requested")
    handler = bridge.ReuseHandler()
    handler.registry, handler.device, handler.environment = registry, 0, {}
    handler.runtime_environment = {"core_limit": [4, 8]}
    handler.capability = SimpleNamespace(compile=False)
    calls, launches = [], []

    def grid(meta):
        calls.append(meta["width"])
        return ((meta["length"] + meta["width"] - 1) // meta["width"], )

    def jit_tail(args, kwargs, launch_grid, *rest, **continuation):
        bound = continuation["bound"]
        actual_grid = continuation["frame"].resolve_grid(launch_grid, bound)
        launches.append((actual_grid, bound["width"]))
        assert actual_grid == expected_grid and bound["width"] == expected_tile
        return "launched"

    assert handler.run(fn, arguments, {}, grid, False, 0, 0, backend, jit_tail) == "launched"
    assert calls == [32] and launches == [(expected_grid, expected_tile)]
