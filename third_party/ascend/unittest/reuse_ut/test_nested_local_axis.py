"""Inner T01 traversal under an unchanged grid-stride row partition."""
from collections import Counter

import pytest
from test_local_axis import source_kernel
from test_tile_rules import tensors as tensor_fixture
from triton.backends.ascend.reuse.analysis import analyze
from triton.backends.ascend.reuse.guards import adapt_tile
from triton.backends.ascend.reuse.local_axis import match_local_axis
from triton.backends.ascend.reuse.model import ParameterClass

tensors = tensor_fixture

NESTED = """
pid = tl.program_id(0)
programs = tl.num_programs(0)
row_tasks = (rows + row_tile - 1) // row_tile
for task in range(pid, row_tasks, programs):
    row = task * row_tile + tl.arange(0, row_tile)
    row_mask = row < rows
    for start in range(0, columns, col_tile):
        column = start + tl.arange(0, col_tile)
        mask = row_mask[:, None] & (column[None, :] < columns)
        address = row[:, None] * stride + column[None, :]
        value = tl.load(src + address, mask=mask, other=0.0)
        result = tl.exp(value.to(tl.float32)).to(value.dtype)
        tl.store(dst + address, result, mask=mask)
"""


def kernel(body=NESTED):
    fn = source_kernel(body, parameters="src, dst, stride, rows, columns, col_tile, row_tile", constexpr=(5, 6))
    for i, (name, p) in enumerate(zip(fn.arg_names, fn.params)):
        p.num, p.name, p.annotation_type, p.do_not_specialize = i, name, "", False
    return fn


def arguments(tensors, rows=17, columns=67, stride=80, col_tile=32, row_tile=4):
    return (tensors(65536,
                    count=rows * stride), tensors(131072,
                                                  count=rows * stride), stride, rows, columns, col_tile, row_tile)


def recipe_for(args, body=NESTED):
    fn = kernel(body)
    return match_local_axis(fn, dict(zip(fn.arg_names, args)))


@pytest.mark.parametrize("grid", [(1, ), (3, ), (257, ), (3, 1, 1)])
@pytest.mark.parametrize("old,new", [(16, 32), (32, 16), (128, 32), (32, 128)])
@pytest.mark.parametrize("rows", [1, 17])
def test_inner_reuse_preserves_all_rows_columns_and_padding(tensors, grid, old, new, rows):
    args = arguments(tensors, rows=rows, col_tile=new)
    recipe = recipe_for(args)
    assert recipe is not None and recipe.tile == 5 and recipe.static_parameters == (6, )
    assert recipe.rule == "MaskedLocalAxisV1" and recipe.grid_axis is None
    actual = adapt_tile(recipe, args, {(5, ): old, (6, ): 4}, grid, 0, compile_mode="simd")
    assert actual == (*args[:5], old, 4)
    visited = Counter(row * args[2] + col
                      for pid in range(grid[0])
                      for task in range(pid, (rows + actual[6] - 1) // actual[6], grid[0])
                      for row in range(task * actual[6], min((task + 1) * actual[6], rows))
                      for start in range(0, args[4], actual[5])
                      for col in range(start, min(start + actual[5], args[4])))
    assert visited == Counter(row * args[2] + col for row in range(rows) for col in range(args[4]))


def test_classification_and_composition_allow_both_tiles(tensors):
    args = arguments(tensors)
    fn = kernel()
    profile = analyze(fn, dict(zip(fn.arg_names, args)), "nested")
    assert [(d.position, d.classification) for d in profile.decisions] == [(5, ParameterClass.ScheduleReusable),
                                                                           (6, ParameterClass.ScheduleReusable)]
    assert adapt_tile(profile.recipe, args, {(5, ): 16, (6, ): 8}, (3, ), 0, compile_mode="simd") == (*args[:5], 16, 8)


@pytest.mark.parametrize("change",
                         ["short_output", "short_input", "overlap", "duplicate_grid", "bad_stride", "overflow"])
def test_full_outer_footprint_guards(tensors, change):
    args = list(arguments(tensors))
    grid = (3, )
    recipe = recipe_for(args)
    if change == "short_output":
        args[1] = tensors(131072, count=16 * 80)
    elif change == "short_input":
        args[0] = tensors(65536, count=16 * 80)
    elif change == "overlap":
        args[1] = tensors(65538, count=17 * 80)
    elif change == "duplicate_grid":
        grid = (3, 2)
    elif change == "bad_stride":
        args[2] = 32
    else:
        args[3] = 2**31 - 1
    assert adapt_tile(recipe, tuple(args), {(5, ): 16, (6, ): 4}, grid, 0, compile_mode="simd") is None


def test_same_mapping_inplace_is_allowed(tensors):
    args = list(arguments(tensors))
    args[1] = args[0]
    assert adapt_tile(recipe_for(args), tuple(args), {(5, ): 16, (6, ): 4}, (3, ), 0, compile_mode="simd")


@pytest.mark.parametrize("old,new", [
    ("rows + row_tile - 1", "rows + col_tile - 1"),
    ("row = task * row_tile", "row = task * row_tile + col_tile"),
    ("result, mask=mask", "result + start, mask=mask"),
    ("result, mask=mask", "result + pid, mask=mask"),
    ("row = task * row_tile", "row = row + task * row_tile"),
    ("column[None, :] < columns", "column[None, :] < columns - 1"),
    ("    row_mask = row < rows", "    row_mask = row < rows\n    if task == 0:\n        break"),
])
def test_unproved_outer_context_or_inner_semantics_decline(tensors, old, new):
    assert recipe_for(arguments(tensors), NESTED.replace(old, new)) is None


def test_memory_effect_after_outer_loop_is_not_checked_with_collapsed_grid(tensors):
    body = NESTED + "\ntl.store(dst + pid, 0.0, mask=pid < programs)\n"
    assert recipe_for(arguments(tensors), body) is None


def test_parameter_names_do_not_select_the_rule(tensors):
    body = NESTED.replace("col_tile", "different_width").replace("row_tile", "fixed_height")
    fn = source_kernel(body, parameters="src, dst, stride, rows, columns, different_width, fixed_height",
                       constexpr=(5, 6))
    recipe = match_local_axis(fn, dict(zip(fn.arg_names, arguments(tensors))))
    assert recipe is not None and recipe.tile == 5


def test_outer_traversal_in_helper_cannot_return_after_first_task(tensors):
    import textwrap

    from test_axis_generalization import helper, with_helpers
    parameters = "src, dst, stride, rows, columns, col_tile, row_tile"
    walk = helper("def walk(" + parameters + "):\n" + textwrap.indent(NESTED.strip(), "    ") + "\n        return\n",
                  constexpr=("col_tile", "row_tile"))
    fn = with_helpers(kernel("walk(" + parameters + ")"), walk=walk)
    assert match_local_axis(fn, dict(zip(fn.arg_names, arguments(tensors)))) is None
