"""Inner tile equivalence under unchanged scalar-loop contexts."""
import ast
from collections import Counter
from types import SimpleNamespace

import pytest
import triton.language as tl
from test_local_axis import source_kernel
from test_tile_rules import tensors as tensor_fixture
from triton.backends.ascend.reuse.analysis import analyze
from triton.backends.ascend.reuse.guards import adapt_tile
from triton.backends.ascend.reuse.local_axis import match_local_axis
from triton.backends.ascend.reuse.model import ParameterClass

tensors = tensor_fixture

BODY = '''
"""An unrelated name and a function docstring must not select the proof."""
pid = tl.program_id(0)
programs = tl.num_programs(0)
chunk = (rows + programs - 1) // programs
begin = pid * chunk
end = tl.minimum(begin + chunk, rows)
for row in range(begin, end):
    scale = tl.load(scales + row)
    for start in range(0, columns, tile):
        col = start + tl.arange(0, tile)
        valid = col < columns
        offset = row * row_stride + col * col_stride
        value = tl.load(data + offset, mask=valid, other=0.0)
        positive = (value > 0) | (value == 0)
        updated = tl.where(positive, value * scale, value / scale)
        updated -= scale
        updated += 1.0
        tl.store(data + offset, updated, mask=valid)
'''


def kernel(body=BODY):
    fn = source_kernel(body, parameters="data, scales, rows, columns, row_stride, col_stride, tile", constexpr=(6, ))
    for i, p in enumerate(fn.params):
        p.num, p.name, p.annotation_type, p.do_not_specialize = i, fn.arg_names[i], "", False
    return fn


def arguments(tensors, rows=7, columns=67, tile=32, strided=False):
    stride = 2 if strided else 1
    pitch = columns * stride + 16
    data = tensors(65536, width=4, count=rows * columns if strided else rows * pitch, contiguous=not strided,
                   storage_count=rows * pitch + 5, storage_offset=5)
    return (data, tensors(131072, width=4, count=rows), rows, columns, pitch, stride, tile)


def recipe(args, body=BODY):
    fn = kernel(body)
    return match_local_axis(fn, dict(zip(fn.arg_names, args)))


@pytest.mark.parametrize("grid", [(1, ), (3, ), (11, ), (3, 1, 1)])
@pytest.mark.parametrize("tiles", [(16, 32), (32, 16), (128, 32), (32, 128)])
@pytest.mark.parametrize("strided", [False, True])
def test_preserved_range_visits_each_requested_element_once(tensors, grid, tiles, strided):
    old, new = tiles
    args = arguments(tensors, tile=new, strided=strided)
    proof = recipe(args)
    assert proof is not None
    adapted = adapt_tile(proof, args, {(6, ): old}, grid, 0, compile_mode="simd")
    assert adapted == (*args[:-1], old)
    rows, columns, pitch, stride = args[2:6]
    chunk = (rows + grid[0] - 1) // grid[0]
    seen = Counter(row * pitch + col * stride
                   for pid in range(grid[0])
                   for row in range(pid * chunk, min((pid + 1) * chunk, rows))
                   for start in range(0, columns, old)
                   for col in range(start, min(start + old, columns)))
    assert seen == Counter(row * pitch + col * stride for row in range(rows) for col in range(columns))


@pytest.mark.parametrize("bounds", [
    ("pid * 3 + 1", "pid * 3 + 3", "1"),
    ("pid * 6", "pid * 6 + 5", "2"),
    ("pid * 3", "tl.maximum(pid * 3, tl.minimum(pid * 3 + 3, rows))", "1"),
])
def test_partial_and_nonunit_outer_domains_need_no_partition_template(tensors, bounds):
    args = arguments(tensors, rows=20)
    body = BODY.replace("range(begin, end)", f"range({', '.join(bounds)})")
    proof = recipe(args, body)
    assert proof is not None
    assert adapt_tile(proof, args, {(6, ): 16}, (3, ), 0, compile_mode="simd")


def test_classification_is_schedule_reusable(tensors):
    args = arguments(tensors)
    fn = kernel()
    profile = analyze(fn, dict(zip(fn.arg_names, args)), "preserved")
    assert [(d.position, d.classification) for d in profile.decisions] == [(6, ParameterClass.ScheduleReusable)]


@pytest.mark.parametrize(
    "change",
    ["storage", "offset", "row_overlap", "zero_stride", "scalar_alias", "grid_duplicates", "overflow", "empty"])
def test_footprint_and_context_guards(tensors, change):
    args = list(arguments(tensors, strided=True))
    proof = recipe(args)
    grid = (3, )
    if change == "storage":
        args[0].storage_count = 100
    elif change == "offset":
        args[0].offset = args[0].storage_count - 1
    elif change == "row_overlap":
        args[4] = 20
    elif change == "zero_stride":
        args[5] = 0
    elif change == "scalar_alias":
        args[1] = tensors(65536, width=4, count=args[2])
    elif change == "grid_duplicates":
        grid = (3, 2)
    elif change == "overflow":
        args[2] = 2**31 - 1
    else:
        args[2] = 0
    assert adapt_tile(proof, tuple(args), {(6, ): 16}, grid, 0, compile_mode="simd") is None


@pytest.mark.parametrize("bounds", ["0, rows", "pid, rows, 0", "pid, rows, -1", "pid, rows, programs + 1"])
def test_unproven_disjointness_and_invalid_steps_decline(tensors, bounds):
    args = arguments(tensors, rows=17)
    proof = recipe(args, BODY.replace("range(begin, end)", f"range({bounds})"))
    assert proof is not None
    assert adapt_tile(proof, args, {(6, ): 16}, (3, ), 0, compile_mode="simd") is None


@pytest.mark.parametrize("old,new", [
    ("chunk = (rows + programs - 1) // programs", "chunk = tile"),
    ("value * scale", "value * tile"),
    ("scale = tl.load(scales + row)", "scale = scale + tl.load(scales + row)"),
    ("updated = tl.where(positive, value * scale, value / scale)", "updated = updated + value"),
    ("updated -= scale", "updated -= start"),
    ("range(begin, end)", "range(begin, tl.load(scales))"),
    ("range(0, columns, tile)", "range(0, columns - 1, tile)"),
    ("data + offset, updated", "data + offset, updated + pid"),
    ("for row in range(begin, end):", "for row in range(begin, end):\n    if row == 1:\n        break"),
])
def test_tile_dependent_effects_and_unproven_state_decline(tensors, old, new):
    assert recipe(arguments(tensors), BODY.replace(old, new)) is None


def test_outer_context_budget_is_a_guard_not_a_tile_classification(tensors, monkeypatch):
    from triton.backends.ascend.reuse import config
    args = arguments(tensors)
    proof = recipe(args)
    monkeypatch.setattr(config, "MAX_OUTER_CONTEXTS", 2)
    assert adapt_tile(proof, args, {(6, ): 16}, (3, ), 0, compile_mode="simd") is None


def test_last_outer_increment_and_canceled_overflow_are_checked(tensors):
    args = arguments(tensors)
    for bounds in ("2147483640, 2147483647, 8", "2147483647 + rows - rows, 2147483647, 1"):
        proof = recipe(args, BODY.replace("range(begin, end)", f"range({bounds})"))
        assert proof is not None
        assert adapt_tile(proof, args, {(6, ): 16}, (1, ), 0, compile_mode="simd") is None


def penalties_kernel():
    from pathlib import Path
    tree = ast.parse(Path(__file__).with_name('_penalties_kernel.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    names = [a.arg for a in fn.args.args]
    fn.decorator_list = []
    return SimpleNamespace(
        arg_names=names, params=tuple(
            SimpleNamespace(name=n, num=i, is_constexpr=n == 'BLOCK_SIZE', annotation_type='', do_not_specialize=n ==
                            'num_seqs') for i, n in enumerate(names)),
        parse=lambda: ast.Module(body=[fn], type_ignores=[]), get_capture_scope=lambda: {'tl': tl})


@pytest.mark.parametrize('strided', [False, True])
@pytest.mark.parametrize('tiles', [(128, 256), (256, 128)])
def test_frozen_penalties_classification_and_full_guard(tensors, strided, tiles):
    fn = penalties_kernel()
    old, new = tiles
    args = arguments(tensors, rows=3, columns=513, tile=new, strided=strided)
    values = [args[0]] + [tensors(262144 + i * 16384, width=4, count=3 * 513 if i < 3 else 3) for i in range(6)]
    values += [3, 513, args[4], args[5], 513, 1, 513, 1, 513, 1, new]
    for tensor in values[:7]:
        tensor.dtype = 'float32'
    bound = dict(zip(fn.arg_names, values))
    profile = analyze(fn, bound, 'frozen-penalties')
    assert profile.decisions[0].classification is ParameterClass.ScheduleReusable
    assert adapt_tile(profile.recipe, tuple(values), {(17, ): old}, (2, ), 0,
                      compile_mode='simd') == (*values[:-1], old)


def test_shape_constraint_alone_is_not_a_semantic_static_proof(tensors):
    args = arguments(tensors)
    fn = kernel(BODY.replace('range(0, columns, tile)', 'range(0, columns - 1, tile)'))
    decision = analyze(fn, dict(zip(fn.arg_names, args)), 'incomplete').decisions[0]
    assert decision.classification is ParameterClass.Unknown
    assert ('TILE_REUSE_UNPROVEN', 0) in decision.reasons


def test_narrow_outer_bound_annotations_require_a_separate_arithmetic_proof(tensors):
    args = arguments(tensors)
    fn = kernel()
    fn.params[2].annotation_type = 'i16'
    assert match_local_axis(fn, dict(zip(fn.arg_names, args))) is None


def test_scalar_load_aliasing_later_rows_is_not_hoisted_into_a_safe_input(tensors):
    args = list(arguments(tensors))
    args[1] = tensors(args[0].data_ptr() + args[4] * 4, width=4, count=args[2])
    assert adapt_tile(recipe(args), tuple(args), {(6, ): 16}, (3, ), 0, compile_mode='simd') is None
