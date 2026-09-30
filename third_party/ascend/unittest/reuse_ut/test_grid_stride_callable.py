"""Callable grids preserve the request while adapting a proved external block count."""
from types import SimpleNamespace

import pytest
from test_tile_rules import kernel
from test_tile_rules import tensors as tensor_fixture
from triton.backends.ascend.reuse.context import InvocationFrame
from triton.backends.ascend.reuse.dispatch import resolve_dispatch
from triton.backends.ascend.reuse.model import JitTailRequest, PreparedLaunch
from triton.backends.ascend.reuse.rules import match_grid_stride

tensors = tensor_fixture


def dispatch_objects(tensors, requested=2048, tiles=(0, 1024), work=None):
    fn = kernel("range(tl.program_id(0), work, tl.num_programs(0))", "block * tile + tl.arange(0, tile)")
    recipe = match_grid_stride(fn)
    assert recipe is not None and not recipe.local_loop and recipe.bound == 3
    work = (5000 + requested - 1) // requested if work is None else work
    arguments = (tensors(65536), tensors(131072), 5000, work, requested)
    types = ("*fp16", "*fp16", "i32", "i32", "constexpr")
    signature = dict(zip(fn.arg_names, types))

    def binder(*args, **kwargs):
        return dict(zip(fn.arg_names,
                        args)), [(ty, value if ty == "constexpr" else None) for ty, value in zip(types, args)], kwargs

    candidates = [
        SimpleNamespace(variant_key=str(tile), source=SimpleNamespace(signature=signature, constants={(4, ): tile},
                                                                      attrs={})) for tile in tiles
    ]
    registry = SimpleNamespace(lookup_exact_ready=lambda *args: None, lookup_family_ready=lambda *args: candidates,
                               has_registration=lambda *args: True, event=lambda *args: None)
    owner = SimpleNamespace(arg_names=fn.arg_names, device_caches={0: (None, None, None, None, binder)})
    spec = SimpleNamespace(owner=owner, variant_key="requested", family_key="family",
                           options=SimpleNamespace(compile_mode="simd"))
    return fn, arguments, SimpleNamespace(recipe=recipe), registry, spec, candidates


@pytest.mark.parametrize("old,requested", [(1024, 2048), (2048, 1024)])
@pytest.mark.parametrize("grid_type", [tuple, list])
def test_grid_stride_callable_uses_request_once_across_candidates(tensors, old, requested, grid_type):
    fn, args, profile, registry, spec, _ = dispatch_objects(tensors, requested, (0, old))
    frame, calls = InvocationFrame(), []

    def grid(meta):
        calls.append((meta["tile"], meta["work"]))
        return grid_type((meta["work"], ))

    result = resolve_dispatch(registry, spec, profile, args, grid, (1, 0), None, 8,
                              resolve_grid=lambda: frame.resolve_grid(grid, dict(zip(fn.arg_names, args))))
    assert isinstance(result, PreparedLaunch)
    assert result.arguments[:3] == args[:3]
    assert result.arguments[3:] == ((5000 + old - 1) // old, old)
    assert result.grid is None  # External block count changes, requested grid does not.
    assert frame.resolve_grid(grid, dict(zip(fn.arg_names, result.arguments))) == grid_type((args[3], ))
    assert calls == [(requested, args[3])]


@pytest.mark.parametrize("rejection", ["partial", "vector_limit", "abi"])
def test_rejected_grid_stride_candidates_keep_request_for_fallback(tensors, rejection):
    fn, args, profile, registry, spec, candidates = dispatch_objects(tensors, tiles=(1024, 512),
                                                                     work=2 if rejection == "partial" else None)
    if rejection == "abi":
        for candidate in candidates:
            candidate.source.signature = {**candidate.source.signature, "a": "*i32"}
    frame, calls = InvocationFrame(), []

    def grid(meta):
        calls.append((meta["tile"], meta["work"]))
        return (3, )

    result = resolve_dispatch(registry, spec, profile, args, grid, (1, 0), None,
                              2 if rejection == "vector_limit" else 8,
                              resolve_grid=lambda: frame.resolve_grid(grid, dict(zip(fn.arg_names, args))))
    assert isinstance(result, JitTailRequest)
    assert frame.resolve_grid(grid, dict(zip(fn.arg_names, args))) == (3, )
    assert args[3:] == (2 if rejection == "partial" else 3, 2048)
    assert calls == [(2048, args[3])]


def test_grid_stride_cold_miss_defers_grid_until_jit_tail(tensors):
    fn, args, profile, registry, spec, _ = dispatch_objects(tensors, tiles=())
    frame, calls = InvocationFrame(), []

    def grid(meta):
        calls.append(meta["tile"])
        return (3, )

    result = resolve_dispatch(registry, spec, profile, args, grid, (1, 0), None, 8,
                              resolve_grid=lambda: frame.resolve_grid(grid, dict(zip(fn.arg_names, args))))
    assert isinstance(result, JitTailRequest) and calls == []
    assert frame.resolve_grid(grid, dict(zip(fn.arg_names, args))) == (3, )
    assert calls == [2048]


def test_grid_stride_grid_error_propagates_without_retry(tensors):
    fn, args, profile, registry, spec, _ = dispatch_objects(tensors)
    frame, calls = InvocationFrame(), []
    error = RuntimeError("grid callback failed")

    def grid(meta):
        calls.append((meta["tile"], meta["work"]))
        raise error

    with pytest.raises(RuntimeError) as caught:
        resolve_dispatch(registry, spec, profile, args, grid, (1, 0), None, 8,
                         resolve_grid=lambda: frame.resolve_grid(grid, dict(zip(fn.arg_names, args))))
    assert caught.value is error and calls == [(2048, 3)]
