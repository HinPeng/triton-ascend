"""Host proofs of element coverage and byte-range guards for cross-tile reuse."""
import ast
import itertools
import sys
from types import SimpleNamespace

import pytest
import triton.language as tl
from triton.backends.ascend.reuse import config
from triton.backends.ascend.reuse.guards import adapt_tile
from triton.backends.ascend.reuse.rules import match_grid_stride


def kernel(loop, index, result="x + length"):
    source = f"""def pointwise(a, out, length, work, tile):
    for block in {loop}:
        index = {index}
        mask = index < length
        x = tl.load(a + index, mask=mask)
        tl.store(out + index, {result}, mask=mask)
"""
    return SimpleNamespace(arg_names=("a", "out", "length", "work", "tile"),
                           params=tuple(SimpleNamespace(is_constexpr=i == 4) for i in range(5)),
                           get_capture_scope=lambda: {"tl": tl}, parse=lambda: ast.parse(source))


@pytest.fixture
def tensors(monkeypatch):

    class Tensor:

        def __init__(self, address, width=2, contiguous=True, count=5000, storage_count=None, storage_offset=0):
            self.address = address
            self.width = width
            self.contiguous = contiguous
            self.count = count
            self.storage_count = count if storage_count is None else storage_count
            self.offset = storage_offset
            self.device = SimpleNamespace(type="npu", index=0)

        def is_contiguous(self):
            return self.contiguous

        def numel(self):
            return self.count

        def untyped_storage(self):
            return SimpleNamespace(nbytes=lambda: self.storage_count * self.width)

        def storage_offset(self):
            return self.offset

        def data_ptr(self):
            return self.address

        def element_size(self):
            return self.width

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(Tensor=Tensor))
    return Tensor


LOOPS = [
    ("range(0, length, tile)", "block + tl.arange(0, tile)", None),
    ("range(tl.cdiv(length, tile))", "block * tile + tl.arange(0, tile)", None),
    ("range(work)", "block * tile + tl.arange(0, tile)", 3),
    ("range(0, work)", "block * tile + tl.arange(0, tile)", 3),
    ("range(0, work, 1)", "block * tile + tl.arange(0, tile)", 3),
    ("range(tl.program_id(0), work, tl.num_programs(0))", "block * tile + tl.arange(0, tile)", 3),
]


@pytest.mark.parametrize("loop,index,bound", LOOPS)
@pytest.mark.parametrize("old_tile,new_tile", [(1024, 1536), (1536, 768), (8192, 1024), (1024, 8192)])
@pytest.mark.parametrize("mode", ["simd", "simd_simt_template"])
def test_loop_reuse_covers_each_element_once(tensors, loop, index, bound, old_tile, new_tile, mode):
    recipe = match_grid_stride(kernel(loop, index))
    assert recipe is not None and recipe.bound == bound
    assert config.supports_tile_reuse(SimpleNamespace(backend="npu"), SimpleNamespace(compile_mode=mode))
    n = 5000
    arguments = (tensors(65536), tensors(131072), n, (n + new_tile - 1) // new_tile, new_tile)
    grids = (1, ) if recipe.local_loop else (1, 3, 9)
    for programs in grids:
        adapted = adapt_tile(recipe, arguments, {(4, ): old_tile}, (programs, ), 0, compile_mode=mode)
        assert adapted is not None and adapted[4] == old_tile
        # Execute the accepted loop/index expressions with ordinary Python lanes.
        # This independently checks the memory coverage after argument adaptation.
        visited = []
        for program in range(programs):
            for lane in range(old_tile):
                scope = {"length": n, "work": adapted[3], "tile": adapted[4], "range": range}
                scope["tl"] = SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b,
                                              arange=lambda start, end, lane=lane: lane,
                                              program_id=lambda axis, program=program: program,
                                              num_programs=lambda axis, programs=programs: programs)
                for block in eval(loop, {"__builtins__": {}}, scope):
                    scope["block"] = block
                    address = eval(index, {"__builtins__": {}}, scope)
                    if address < n:
                        visited.append(address)
        assert sorted(visited) == list(range(n))
        if bound is None:
            assert adapted[3] == arguments[3]


@pytest.mark.parametrize("loop,index,bound", LOOPS[:-1])
def test_single_program_rule_rejects_duplicate_writes(tensors, loop, index, bound):
    recipe = match_grid_stride(kernel(loop, index))
    args = (tensors(65536), tensors(131072), 5000, 4, 1536)
    assert adapt_tile(recipe, args, {(4, ): 1024}, (2, ), 0, compile_mode="simd") is None


def outer_kernel(loop, index, input_base, output_base=None):
    fn = kernel(loop, index, result="x + tl.program_id(1)")
    source = ast.unparse(fn.parse()).replace("a + index", f"a + ({input_base}) + index")
    source = source.replace("out + index", f"out + ({output_base or input_base}) + index")
    fn.parse = lambda: ast.parse(source)
    return fn


@pytest.mark.parametrize("grid", [(12, ), (2, 3), (2, 3, 2)])
@pytest.mark.parametrize("loop,index,bound", LOOPS[:-1])
def test_local_axis_is_independent_of_outer_grid(tensors, grid, loop, index, bound):
    base = "((tl.program_id(2) * tl.num_programs(1) + tl.program_id(1)) * tl.num_programs(0) + tl.program_id(0)) * length"
    recipe = match_grid_stride(outer_kernel(loop, index, base))
    assert recipe is not None and recipe.local_loop
    dimensions = (*grid, *((1, ) * (3 - len(grid))))
    count = 5000
    for size in grid:
        count *= size
    args = (tensors(65536, count=count), tensors(1048576, count=count), 5000, 4, 1536)
    # No vector-core limit applies to a tile axis internal to each program.
    adapted = adapt_tile(recipe, args, {(4, ): 1024}, grid, 0, compile_mode="simd")
    assert adapted is not None and adapted[2:3] == args[2:3] and adapted[4] == 1024
    visited = []
    for ids in itertools.product(*(range(size) for size in dimensions)):
        origin = ((ids[2] * dimensions[1] + ids[1]) * dimensions[0] + ids[0]) * 5000
        for start in range(0, 5000, adapted[4]):
            visited.extend(origin + start + lane for lane in range(adapted[4]) if start + lane < 5000)
    assert sorted(visited) == list(range(count))


@pytest.mark.parametrize("base", ["tl.program_id(0) * tile", "tl.program_id(0) * work"])
def test_outer_partition_cannot_depend_on_adapted_arguments(base):
    assert match_grid_stride(outer_kernel(*LOOPS[2][:2], base)) is None


def test_local_layout_guards_cover_the_whole_grid(tensors):
    recipe = match_grid_stride(outer_kernel(*LOOPS[0][:2], "tl.program_id(0) * length"))
    args = (tensors(65536, count=10000), tensors(131072, count=5000), 5000, 0, 1536)
    assert adapt_tile(recipe, args, {(4, ): 1024}, (2, ), 0, compile_mode="simd") is None


def test_same_envelope_with_different_program_mapping_is_not_inplace(tensors):
    row_major = "(tl.program_id(0) * 2 + tl.program_id(1)) * length"
    column_major = "(tl.program_id(1) * 2 + tl.program_id(0)) * length"
    recipe = match_grid_stride(outer_kernel(*LOOPS[0][:2], row_major, column_major))
    tensor = tensors(65536, count=20000)
    args = (tensor, tensor, 5000, 0, 1536)
    assert adapt_tile(recipe, args, {(4, ): 1024}, (2, 2), 0, compile_mode="simd") is None


@pytest.mark.parametrize("loop,index", [
    ("range(1, work)", "block * tile + tl.arange(0, tile)"),
    ("range(0, work, 2)", "block * tile + tl.arange(0, tile)"),
    ("range(0, length, tile)", "block * tile + tl.arange(0, tile)"),
    ("range(tl.cdiv(work, tile))", "block * tile + tl.arange(0, tile)"),
])
def test_incomplete_or_different_coverage_is_rejected(loop, index):
    assert match_grid_stride(kernel(loop, index)) is None


@pytest.mark.parametrize("loop,index,bound", LOOPS)
def test_schedule_dependent_results_are_rejected(loop, index, bound):
    assert match_grid_stride(kernel(loop, index, result="x + block")) is None
    assert match_grid_stride(kernel(loop, index, result="x + tile")) is None
    if bound is not None:
        assert match_grid_stride(kernel(loop, index, result="x + work")) is None


@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_element_widths_and_alias_guards(tensors, width):
    recipe = match_grid_stride(kernel(*LOOPS[0][:2]))
    a = tensors(65536, width)

    def adapt(out):
        return adapt_tile(recipe, (a, out, 5000, 0, 1536), {(4, ): 1024}, (1, ), 0, compile_mode="simd")

    assert adapt(tensors(131072, width)) is not None
    assert adapt(a) is not None
    assert adapt(tensors(65536 + width, width)) is None
    assert adapt(tensors(65536, width * 2)) is None
    assert adapt(tensors(131072, width, contiguous=False)) is None


def test_wrong_external_tile_count_is_rejected(tensors):
    recipe = match_grid_stride(kernel(*LOOPS[2][:2]))
    args = (tensors(65536), tensors(131072), 5000, 3, 1536)
    assert adapt_tile(recipe, args, {(4, ): 1024}, (1, ), 0, compile_mode="simd") is None


@pytest.mark.parametrize("loop,index,bound", LOOPS)
def test_loop_carried_state_is_rejected(loop, index, bound):
    fn = kernel(loop, index)
    tree = fn.parse()
    loop_node = tree.body[0].body[0]
    loop_node.body.insert(0, ast.parse("state = state + 1").body[0])
    fn.parse = lambda: tree
    assert match_grid_stride(fn) is None


@pytest.mark.parametrize("backend", ["npu", "cuda"])
@pytest.mark.parametrize("mode", ["simd", "simd_simt_template", "simt_only"])
def test_target_capability_excludes_pure_simt(backend, mode):
    assert config.supports_tile_reuse(SimpleNamespace(backend=backend),
                                      SimpleNamespace(compile_mode=mode)) is (backend == "npu" and mode != "simt_only")


@pytest.mark.parametrize("programs", [65, 257])
@pytest.mark.parametrize("old,requested", [(16, 32), (32, 16)])
@pytest.mark.parametrize("mode", ["simd", "simd_simt_template"])
def test_grid_stride_large_grid_covers_each_element_once(tensors, programs, old, requested, mode):
    recipe = match_grid_stride(kernel(*LOOPS[-1][:2]))
    n = 5000
    args = (tensors(65536), tensors(131072), n, (n + requested - 1) // requested, requested)
    adapted = adapt_tile(recipe, args, {(4, ): old}, (programs, 1, 1), 0, compile_mode=mode)
    assert adapted is not None and adapted[3:] == ((n + old - 1) // old, old)
    visited = [
        block * old + lane
        for pid in range(programs)
        for block in range(pid, adapted[3], programs)
        for lane in range(old)
        if block * old + lane < n
    ]
    assert sorted(visited) == list(range(n))
