"""Independent D/T eligibility and tile adaptation across legal tile sizes."""
import sys
from types import SimpleNamespace

import pytest
import triton
import triton.language as tl

from test_analysis import Pointer, decision
from triton.backends.ascend.reuse import config
from triton.backends.ascend.reuse.analysis import analyze
from triton.backends.ascend.reuse.binding import ArgumentBinder
from triton.backends.ascend.reuse.guards import adapt_tile
from triton.backends.ascend.reuse.identity import family_key
from triton.backends.ascend.reuse.model import LaunchRecipe, ParameterClass


@triton.jit
def integer_pointwise(a, out, length: tl.constexpr, work, tile: tl.constexpr):
    p = tl.program_id(0)
    step = tl.num_programs(0)
    for block in range(p, work, step):
        index = block * tile + tl.arange(0, tile)
        mask = index < length
        x = tl.load(a + index, mask=mask)
        tl.store(out + index, x + length, mask=mask)


@pytest.mark.parametrize("arch,mode,tile_allowed", [
    ("Ascend910B", "simd", True),
    ("Ascend910B", "simd_simt_template", True),
    ("Ascend950PR_9579", "simd", True),
    ("Ascend950PR_9579", "simt_only", False),
    ("Ascend950PR_9579", "simd_simt_template", True),
])
def test_dynamic_profile_survives_tile_target_rejection(arch, mode, tile_allowed):
    target = SimpleNamespace(backend="npu", arch=arch)
    options = SimpleNamespace(compile_mode=mode)
    assert config.supports_target(target)
    assert config.supports_tile_reuse(target, options) is tile_allowed
    bound, _ = ArgumentBinder(integer_pointwise.signature)(Pointer(), Pointer(), 5000, 5, 1024)
    profile = analyze(integer_pointwise, bound, integer_pointwise.cache_key,
                      allow_tile_reuse=config.supports_tile_reuse(target, options))
    assert profile.plan.dynamic == (2, )
    assert decision(profile, 2).classification is ParameterClass.RuntimeEligible
    assert (profile.recipe is not None) is tile_allowed
    tile = decision(profile, 4)
    if tile_allowed:
        assert tile.classification is ParameterClass.ScheduleReusable
    else:
        assert tile.classification is ParameterClass.Unknown
        assert "TILE_REUSE_TARGET_UNSUPPORTED" in repr(tile.reasons)

    def family(size):
        source = SimpleNamespace(signature={"length": "i32"}, constants={(4, ): size}, attrs={})
        return family_key(profile, source, options, target, {}, options_hash="options")

    # D-only families must retain the tile in their identity, including mixed D/T kernels.
    assert (family(1024) == family(1536)) is tile_allowed


def test_foreign_backend_stays_outside_both_paths():
    target = SimpleNamespace(backend="cuda", arch="Ascend950PR_9579")
    options = SimpleNamespace(compile_mode="simd_simt_template")
    assert not config.supports_target(target)
    assert not config.supports_tile_reuse(target, options)


@pytest.fixture
def tensor_metadata(monkeypatch):
    dtype = object()

    class Tensor:

        def __init__(self, address):
            self.address = address
            self.dtype = dtype
            self.device = SimpleNamespace(type="npu", index=0)

        def is_contiguous(self):
            return True

        def numel(self):
            return 5000

        def data_ptr(self):
            return self.address

        def element_size(self):
            return 4

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(Tensor=Tensor, float32=dtype))
    return Tensor(65536), Tensor(131072)


@pytest.mark.parametrize("old_tile,new_tile,simt_legal", [(1024, 1536, False), (1536, 1024, False), (1536, 768, False),
                                                          (1024, 2048, True)])
@pytest.mark.parametrize("mode", ["simd", "simd_simt_template", "simt_only"])
def test_tile_adaptation_preserves_exact_coverage(tensor_metadata, old_tile, new_tile, simt_legal, mode):
    recipe = LaunchRecipe(tile=4, extent=2, bound=3, pointers=(0, 1), stores=(1, ))
    arguments = (*tensor_metadata, 5000, triton.cdiv(5000, new_tile), new_tile)
    adapted = adapt_tile(recipe, arguments, {(4, ): old_tile}, (3, ), 0, compile_mode=mode)
    if mode == "simt_only" and not simt_legal:
        assert adapted is None
        return
    assert adapted is not None and adapted[4] == old_tile
    # Enumerate the rule's actual program/loop/lane mapping: each element exactly once.
    visited = [
        block * adapted[4] + lane
        for program in range(3)
        for block in range(program, adapted[3], 3)
        for lane in range(adapted[4])
        if block * adapted[4] + lane < 5000
    ]
    assert sorted(visited) == list(range(5000))
    assert arguments[4] == new_tile


@pytest.mark.parametrize("grid,blocks", [((0, ), 4), ((2**31, ), 4), ((True, ), 4), ((3, 2), 4), ((3, ), 1)])
def test_non_power_of_two_support_keeps_launch_guards(tensor_metadata, grid, blocks):
    recipe = LaunchRecipe(tile=4, extent=2, bound=3, pointers=(0, 1), stores=(1, ))
    arguments = (*tensor_metadata, 5000, blocks, 1536)
    assert adapt_tile(recipe, arguments, {(4, ): 1024}, grid, 0, compile_mode="simd_simt_template") is None


@pytest.mark.parametrize("arch,mode,tile_allowed", [("Ascend910B", "simd", True),
                                                    ("Ascend950PR_9579", "simt_only", False),
                                                    ("Ascend950PR_9579", "simd_simt_template", True)])
@pytest.mark.parametrize("grid_type", [tuple, list])
@pytest.mark.parametrize("ready", [False, True])
@pytest.mark.parametrize("length", [5000, 2**40, 0.5])
def test_bridge_keeps_dynamicization_and_shares_grid_with_launch(monkeypatch, arch, mode, tile_allowed, grid_type,
                                                                 ready, length):
    from triton._C import libtriton
    from triton.backends.ascend.reuse import bridge, dispatch, identity, protocol, transform
    from triton.backends.ascend.reuse.model import JitTailRequest, PreparedLaunch

    target = SimpleNamespace(backend="npu", arch=arch)
    options = SimpleNamespace(compile_mode=mode, ir_override=None, hash=lambda: "options")
    backend = SimpleNamespace(target=target, parse_options=lambda kwargs: options)
    monkeypatch.setitem(sys.modules, "triton.backends.ascend.compiler",
                        SimpleNamespace(NPUOptions=SimpleNamespace(__dataclass_fields__={})))
    monkeypatch.setattr(bridge, "requires_original", lambda *args: False)
    monkeypatch.setattr(libtriton, "get_cache_invalidating_env_vars", lambda: {})
    monkeypatch.setattr(protocol, "environment", lambda *args: {"core_limit": [4, 8]})
    monkeypatch.setattr(identity, "variant_key", lambda *args: "variant")

    names = integer_pointwise.arg_names
    signature = {"a": "*i32", "out": "*i32", "length": "i32", "work": "i32", "tile": "constexpr"}

    def specialize(*args, **kwargs):
        return dict(zip(names, args)), [(signature[n], v if n == "tile" else None) for n, v in zip(names, args)], kwargs

    def pack(actual_backend, kwargs, bound, specialization, raw):
        return options, signature, {(4, ): bound["tile"]}, {}

    owner = SimpleNamespace(
        arg_names=names, device_caches={0: ({}, {}, target, backend, specialize)},
        _pack_args=pack, ASTSource=lambda fn, signature, constants, attrs: SimpleNamespace(
            signature=signature, constants=constants, attrs=attrs))
    monkeypatch.setattr(transform.FrozenJITFunction, "from_plan", lambda plan: owner)
    observed = {}
    selected_kernel = object()

    def resolve(registry, spec, profile, arguments, grid, *args, resolve_grid=None):
        assert profile.plan.dynamic == (2, )
        assert arguments[2] == length
        assert (profile.recipe is not None) is tile_allowed
        assert type(grid) is tuple and grid == (3, )
        observed["grid"] = grid
        executable = SimpleNamespace(kernel=selected_kernel, variant_key=spec.variant_key)
        return PreparedLaunch(executable, arguments) if ready else JitTailRequest(spec)

    monkeypatch.setattr(dispatch, "resolve_dispatch", resolve)

    def launch(args, kwargs, grid, *positional, **keywords):
        assert keywords["owner"] is owner  # the original fallback does not pass an owner
        assert keywords["bound"]["length"] == length
        assert grid is observed["grid"]
        if ready:
            assert keywords["prepared"] is selected_kernel
        return selected_kernel

    grid = grid_type((3, ))
    result = bridge.ReuseHandler().run(integer_pointwise, (Pointer(), Pointer(), length, 4, 1536),
                                       {"compile_mode": mode}, grid, False, 0, 0, backend, launch)
    assert result is selected_kernel
    assert grid == grid_type((3, ))
