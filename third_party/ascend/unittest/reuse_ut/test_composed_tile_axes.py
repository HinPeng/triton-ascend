"""Independent row/column proofs, candidate composition and full ABI dispatch."""
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_nested_local_axis import NESTED, arguments, kernel
from test_tile_rules import tensors as tensor_fixture
from triton.backends.ascend.reuse.analysis import analyze
from triton.backends.ascend.reuse.dispatch import resolve_dispatch
from triton.backends.ascend.reuse.guards import adapt_tile
from triton.backends.ascend.reuse.identity import family_key
from triton.backends.ascend.reuse.local_axis import LocalAxis, match_tile_axes
from triton.backends.ascend.reuse.model import ComposedTileRecipe, JitTailRequest, PreparedLaunch

tensors = tensor_fixture


def profile(args, body=NESTED):
    fn = kernel(body)
    return analyze(fn, dict(zip(fn.arg_names, args)), "two-axes")


@pytest.mark.parametrize("grid", [(1, ), (3, ), (257, ), (3, 1, 1)])
@pytest.mark.parametrize("old_m,new_m", [(1, 4), (4, 1), (8, 4), (4, 8)])
@pytest.mark.parametrize("old_n,new_n", [(16, 32), (32, 16), (128, 32)])
def test_both_tiles_cover_each_logical_element_once(tensors, grid, old_m, new_m, old_n, new_n):
    args = arguments(tensors, row_tile=new_m, col_tile=new_n)
    proof = profile(args).recipe
    assert isinstance(proof, ComposedTileRecipe) and proof.tiles == (5, 6)
    adapted = adapt_tile(proof, args, {(5, ): old_n, (6, ): old_m}, grid, 0, compile_mode="simd")
    assert adapted == (*args[:5], old_n, old_m)
    visited = Counter(row * args[2] + col
                      for pid in range(grid[0])
                      for task in range(pid, (args[3] + old_m - 1) // old_m, grid[0])
                      for row in range(task * old_m, min((task + 1) * old_m, args[3]))
                      for start in range(0, args[4], old_n)
                      for col in range(start, min(start + old_n, args[4])))
    assert visited == Counter(r * args[2] + c for r in range(args[3]) for c in range(args[4]))


def test_each_axis_still_has_an_independent_whole_kernel_proof(tensors):
    args = arguments(tensors)
    fn = kernel()
    row = LocalAxis(fn, 6, dict(zip(fn.arg_names, args))).run()
    assert row.grid_stride_axis == 0 and row.static_parameters == (5, )
    assert adapt_tile(row, args, {(5, ): 32, (6, ): 8}, (3, ), 0, compile_mode="simd") == (*args[:6], 8)
    assert adapt_tile(row, args, {(5, ): 16, (6, ): 8}, (3, ), 0, compile_mode="simd") is None


def test_final_shape_limit_cannot_be_bypassed_by_independent_axis_checks(tensors):
    args = arguments(tensors, row_tile=1, col_tile=16)
    proof = profile(args).recipe
    assert adapt_tile(proof, args, {(5, ): 16384, (6, ): 1}, (3, ), 0, compile_mode="simd")
    assert adapt_tile(proof, args, {(5, ): 16, (6, ): 128}, (3, ), 0, compile_mode="simd")
    assert adapt_tile(proof, args, {(5, ): 16384, (6, ): 128}, (3, ), 0, compile_mode="simd") is None


def test_order_dependent_substitution_declines(tensors):
    args = arguments(tensors, row_tile=128, col_tile=16)
    proof = profile(args).recipe
    constants = {(5, ): 16384, (6, ): 1}
    assert adapt_tile(proof.recipes[0], args, {(5, ): 16384, (6, ): 128}, (3, ), 0, compile_mode="simd") is None
    assert adapt_tile(proof, args, constants, (3, ), 0, compile_mode="simd") is None
    # Reversing the proof order must not rescue the coupled request.
    reversed_proof = replace(proof, recipes=tuple(reversed(proof.recipes)))
    assert adapt_tile(reversed_proof, args, constants, (3, ), 0, compile_mode="simd") is None


@pytest.mark.parametrize("change", ["overlap", "short", "duplicate_grid", "missing_tile", "invalid_tile", "simt"])
def test_composition_rejects_invalid_final_candidate(tensors, change):
    args = list(arguments(tensors))
    proof = profile(args).recipe
    grid, constants, mode = (3, ), {(5, ): 16, (6, ): 8}, "simd"
    if change == "overlap":
        args[1] = tensors(65538, count=17 * 80)
    elif change == "short":
        args[1] = tensors(131072, count=16 * 80)
    elif change == "duplicate_grid":
        grid = (3, 2)
    elif change == "missing_tile":
        del constants[(6, )]
    elif change == "invalid_tile":
        constants[(6, )] = 0
    else:
        mode = "simt_only"
    assert adapt_tile(proof, tuple(args), constants, grid, 0, compile_mode=mode) is None


def test_specialized_control_is_not_relaxed_as_a_partition(tensors):
    args = arguments(tensors)
    proof = profile(args).recipe
    # A partition parameter may also be a specialized branch selector. It
    # cannot be rewritten during composition under the original branch proof.
    column = replace(proof.recipes[0], control_parameters=(6, ))
    protected = replace(proof, recipes=(column, proof.recipes[1]))
    assert adapt_tile(protected, args, {(5, ): 16, (6, ): 8}, (3, ), 0, compile_mode="simd") is None


def test_family_omits_both_tiles_but_retains_other_constants(tensors):
    p = profile(arguments(tensors))
    options = SimpleNamespace(hash=lambda: "options")

    def key(n, m, tag):
        source = SimpleNamespace(signature={"src": "*fp32"}, constants={(5, ): n, (6, ): m, (7, ): tag}, attrs={})
        return family_key(p, source, options, "target", {})

    assert key(16, 1, True) == key(32, 8, True)
    assert key(16, 1, True) != key(32, 8, False)


@pytest.mark.parametrize("body", [
    NESTED.replace("rows + row_tile - 1", "rows + row_tile - 1 + pid"),
    NESTED.replace("result, mask=mask", "result + task, mask=mask"),
    NESTED.replace("range(pid, row_tasks, programs)", "range(pid, row_tasks, programs + 1)"),
    NESTED.replace("row_mask = row < rows", "row_mask = row < rows - 1"),
])
def test_row_proof_declines_changed_partition_or_result(tensors, body):
    fn = kernel(body)
    proof = match_tile_axes(fn, dict(zip(fn.arg_names, arguments(tensors))))
    assert proof is None or 6 not in proof.tiles


@pytest.mark.parametrize("grid_kind", ["tuple", "list", "callable"])
def test_dispatch_selects_whole_candidate_and_checks_both_constants(tensors, grid_kind):
    args = arguments(tensors)
    p = profile(args)
    fn = kernel()
    types = ["*fp32", "*fp32", "i32", "i32", "i32", "constexpr", "constexpr"]
    signature = dict(zip(fn.arg_names, types))

    def binder(*actual, **options):
        return {}, [(kind, value if kind == "constexpr" else None) for kind, value in zip(types, actual)], {}

    candidates = [
        SimpleNamespace(variant_key=str(i), source=SimpleNamespace(signature=signature, constants={(5, ): n, (6, ): m},
                                                                   attrs={}))
        for i, (n, m) in enumerate(((16384, 128), (16, 8)))
    ]
    events, calls = [], []
    registry = SimpleNamespace(lookup_exact_ready=lambda *a: None, lookup_family_ready=lambda *a: candidates,
                               has_registration=lambda *a: True, event=events.append)
    owner = SimpleNamespace(arg_names=fn.arg_names, device_caches={0: (None, None, None, None, binder)})
    spec = SimpleNamespace(owner=owner, variant_key="requested", family_key="family",
                           options=SimpleNamespace(compile_mode="simd"))

    def resolve():
        calls.append(1)
        return [3]

    grid = (3, ) if grid_kind == "tuple" else [3] if grid_kind == "list" else lambda meta: [3]
    result = resolve_dispatch(registry, spec, p, args, grid, (1, 0), None, resolve_grid=resolve)
    assert isinstance(result, PreparedLaunch) and result.executable is candidates[1]
    assert result.arguments == (*args[:5], 16, 8) and result.grid is None
    assert "candidate_rejected" in events and "ready_compatible_hit" in events
    candidates.pop()
    fallback = resolve_dispatch(registry, spec, p, args, grid, (1, 0), None, resolve_grid=resolve)
    assert isinstance(fallback, JitTailRequest)
    assert args[5:] == (32, 4)


def test_only_changed_axis_needs_a_replacement_proof(tensors):
    args = arguments(tensors)
    proof = profile(args).recipe
    assert adapt_tile(proof, args, {(5, ): 32, (6, ): 8}, (3, ), 0, compile_mode="simd") == (*args[:6], 8)
    assert adapt_tile(proof, args, {(5, ): 16, (6, ): 4}, (3, ), 0, compile_mode="simd") == (*args[:5], 16, 4)


@pytest.mark.parametrize("dependency", ["extent", "axis_extent", "control", "static"])
def test_cross_axis_dependencies_decline_at_admission_and_dispatch(tensors, monkeypatch, dependency):
    from triton.backends.ascend.reuse import local_axis
    args = arguments(tensors)
    proof = profile(args).recipe
    first = proof.recipes[0]
    peer = proof.recipes[1].tile
    if dependency == "extent":
        first = replace(first, extent=("mul", first.extent, ("arg", peer)))
    elif dependency == "axis_extent":
        first = replace(first, axes=(replace(first.axes[0], extent=("arg", peer)), ))
    elif dependency == "control":
        first = replace(first, control_parameters=(peer, ))
    else:
        first = replace(first, static_parameters=(peer, ), partition_parameters=())
    coupled = replace(proof, recipes=(first, *proof.recipes[1:]))
    monkeypatch.setattr(local_axis, "local_axis_recipes", lambda *args: iter(coupled.recipes))
    assert not isinstance(match_tile_axes(kernel(), {}), ComposedTileRecipe)
    assert adapt_tile(coupled, args, {(5, ): 16, (6, ): 8}, (3, ), 0, compile_mode="simd") is None
    assert args[5:] == (32, 4)


def test_concrete_peer_tile_proofs_are_not_enough(tensors, monkeypatch):
    from triton.backends.ascend.reuse import local_axis
    args = arguments(tensors)
    proof = profile(args).recipe
    fn = kernel()
    seen = []

    def recipes(fn, bound):
        seen.append(dict(bound))
        return iter(proof.recipes if "row_tile" in bound else ())

    monkeypatch.setattr(local_axis, "local_axis_recipes", recipes)
    assert not isinstance(match_tile_axes(fn, dict(zip(fn.arg_names, args))), ComposedTileRecipe)
    assert "row_tile" not in seen[1] and "col_tile" not in seen[1]
    assert seen[1]["rows"] == args[3] and seen[1]["src"] is args[0]


@pytest.mark.parametrize("extent", ["columns + row_tile", "columns * row_tile"])
def test_logical_domain_depending_on_peer_tile_is_not_composed(tensors, extent):
    args = arguments(tensors)
    body = NESTED.replace("range(0, columns, col_tile)", f"range(0, {extent}, col_tile)")
    body = body.replace("column[None, :] < columns", f"column[None, :] < {extent}")
    assert not isinstance(profile(args, body).recipe, ComposedTileRecipe)


def test_grid_stride_proof_does_not_hide_suffix_memory_effects(tensors):
    body = NESTED + "\ntl.store(dst, 0.0, mask=True)\n"
    fn = kernel(body)
    assert match_tile_axes(fn, dict(zip(fn.arg_names, arguments(tensors)))) is None


def test_fixed_inner_loop_in_helper_cannot_return_early(tensors):
    import textwrap
    from test_axis_generalization import helper, with_helpers
    cut = NESTED.index("    for start")
    body = textwrap.dedent(NESTED[cut:])
    inner = helper(
        "def walk(src, dst, row, row_mask, stride, columns, col_tile):\n" + textwrap.indent(body, "    ") +
        "        return\n", constexpr=("col_tile", ))
    fn = with_helpers(kernel(NESTED[:cut] + "    walk(src, dst, row, row_mask, stride, columns, col_tile)\n"),
                      walk=inner)
    proof = match_tile_axes(fn, dict(zip(fn.arg_names, arguments(tensors))))
    assert proof is None or 6 not in proof.tiles


def test_composition_preserves_silu_values_and_padding():
    import ast
    import numpy as np

    class Tile(np.ndarray):

        def to(self, dtype):
            return self.astype(dtype)

    class Pointer:

        def __init__(self, data, offsets=0):
            self.data, self.offsets = data, offsets

        def __add__(self, offsets):
            return Pointer(self.data, self.offsets + offsets)

    def load(pointer, mask, other=0):
        offsets, mask = np.broadcast_arrays(pointer.offsets, mask)
        value = np.full(offsets.shape, other, dtype=pointer.data.dtype)
        value[mask] = pointer.data.reshape(-1)[offsets[mask]]
        return value.view(Tile)

    def store(pointer, value, mask):
        offsets, value, mask = np.broadcast_arrays(pointer.offsets, value, mask)
        pointer.data.reshape(-1)[offsets[mask]] = value[mask]

    body = NESTED.replace("tl.exp(value.to(tl.float32)).to(value.dtype)",
                          "(value.to(tl.float32) / (1 + tl.exp(-value.to(tl.float32)))).to(value.dtype)")
    fn = kernel(body)
    functions = SimpleNamespace(arange=lambda lo, hi: np.arange(lo, hi).view(Tile), exp=np.exp, float32=np.float32,
                                load=load, store=store)
    scope = {"tl": functions}
    exec(compile(fn.parse(), "nested_silu_host.py", "exec"), scope)
    for dtype in (np.float16, np.float32):
        data = np.random.default_rng(5).normal(size=(17, 80)).astype(dtype)
        expected = data[:, :67].astype(np.float32)
        expected = (expected / (1 + np.exp(-expected))).astype(dtype)
        for m, n in ((1, 16), (4, 32), (8, 128), (32, 16)):
            for programs in (1, 3, 25):
                output = np.full_like(data, -123)
                functions.num_programs = lambda axis: programs
                for pid in range(programs):
                    functions.program_id = lambda axis: pid
                    scope['arbitrary_name'](Pointer(data), Pointer(output), 80, 17, 67, n, m)
                np.testing.assert_array_equal(output[:, :67], expected)
                assert (output[:, 67:] == -123).all()


def test_parameter_order_does_not_choose_which_axis_can_be_reused(tensors):
    fn = kernel()
    args = arguments(tensors)
    tree = fn.parse()
    order = (6, 5, 0, 1, 2, 3, 4)
    tree.body[0].args.args = [tree.body[0].args.args[i] for i in order]
    fn.arg_names = tuple(fn.arg_names[i] for i in order)
    fn.params = tuple(fn.params[i] for i in order)
    for i, p in enumerate(fn.params):
        p.num = i
    fn.parse = lambda: tree
    actual = tuple(args[i] for i in order)
    p = analyze(fn, dict(zip(fn.arg_names, actual)), "reordered")
    assert p.recipe.tiles == (0, 1)
    assert adapt_tile(p.recipe, actual, {(0, ): 8, (1, ): 16}, (3, ), 0, compile_mode="simd") == (8, 16, *actual[2:])


@pytest.mark.parametrize("old,new", [
    ("range(pid, row_tasks, programs)", "range(pid + 2147483647 - 2147483647, row_tasks, programs)"),
    ("range(pid, row_tasks, programs)", "range(pid, row_tasks, programs + 2147483647 - 2147483647)"),
    ("range(0, columns, col_tile)", "range(0, columns, col_tile + 2147483647 - 2147483647)"),
])
@pytest.mark.parametrize("candidate", [{(5, ): 16, (6, ): 4}, {(5, ): 32, (6, ): 8}, {(5, ): 16, (6, ): 8}])
def test_loop_boundary_cancellation_does_not_hide_overflow(tensors, old, new, candidate):
    args = arguments(tensors)
    p = profile(args, NESTED.replace(old, new))
    assert p.recipe is not None
    assert adapt_tile(p.recipe, args, candidate, (3, ), 0, compile_mode="simd") is None


def test_independent_proofs_do_not_specialize_peer_values(tensors):
    first = profile(arguments(tensors, col_tile=16, row_tile=1)).recipe
    second = profile(arguments(tensors, col_tile=128, row_tile=8)).recipe
    assert isinstance(first, ComposedTileRecipe)
    assert first == second
