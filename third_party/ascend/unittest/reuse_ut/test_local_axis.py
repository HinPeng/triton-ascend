"""The same local-axis proof for unrelated pointwise and reduction kernels."""
import ast
import textwrap
from types import SimpleNamespace

import pytest
import triton.language as tl
from test_tile_rules import tensors as tensor_fixture
from triton.backends.ascend.reuse.guards import adapt_tile
from triton.backends.ascend.reuse.local_axis import match_local_axis

tensors = tensor_fixture


def source_kernel(body, parameters="src, dst, length, width", constexpr=(3, )):
    source = "def arbitrary_name(" + parameters + "):\n" + textwrap.indent(textwrap.dedent(body).strip(), "    ")
    names = tuple(n.strip() for n in parameters.split(","))
    return SimpleNamespace(arg_names=names,
                           params=tuple(SimpleNamespace(is_constexpr=i in constexpr) for i in range(len(names))),
                           parse=lambda: ast.parse(source), get_capture_scope=lambda: {"tl": tl})


POINTWISE = """
base = tl.program_id(0) * length
for start in range(0, length, width):
    index = start + tl.arange(0, width)
    valid = index < length
    value = tl.load(src + base + index, mask=valid, other=0.0)
    result = tl.exp(value) + value * value
    tl.store(dst + base + index, result, mask=valid)
"""

SUM = """
base = tl.program_id(0) * length
state = tl.zeros((width,), dtype=tl.float32)
for start in range(0, length, width):
    index = start + tl.arange(0, width)
    valid = index < length
    value = tl.load(src + base + index, mask=valid, other=0.0)
    state += value * value
total = tl.sum(state, axis=0)
tl.store(dst + tl.program_id(0), tl.sqrt(total), mask=tl.program_id(0) < tl.num_programs(0))
"""

MAXIMUM = """
base = tl.program_id(0) * length
state = tl.full((width,), -float("inf"), dtype=tl.float32)
for start in range(0, length, width):
    index = start + tl.arange(0, width)
    valid = index < length
    value = tl.load(src + base + index, mask=valid, other=-float("inf"))
    state = tl.maximum(state, value)
total = tl.max(state, axis=0)
tl.store(dst + tl.program_id(0), total, mask=tl.program_id(0) < tl.num_programs(0))
"""

MINIMUM = MAXIMUM.replace("-float(\"inf\")", "float(\"inf\")").replace("tl.maximum",
                                                                       "tl.minimum").replace("tl.max(", "tl.min(")


@pytest.mark.parametrize("body", [POINTWISE, SUM, MAXIMUM, MINIMUM])
def test_same_pattern_accepts_unrelated_computations(tensors, body):
    recipe = match_local_axis(source_kernel(body))
    assert recipe is not None and recipe.tile == 3 and recipe.rule == "MaskedLocalAxisV1"
    args = (tensors(65536, count=201), tensors(131072, count=201), 67, 32)
    adapted = adapt_tile(recipe, args, {(3, ): 16}, (3, ), 0, None, compile_mode="simd")
    assert adapted is not None and adapted[:3] == args[:3] and adapted[3] == 16


def test_parameter_order_and_local_names_are_not_contracts():
    fn = source_kernel(
        POINTWISE.replace("width", "CHUNK").replace("index", "column"), parameters="CHUNK, dst, length, src",
        constexpr=(0, ))
    recipe = match_local_axis(fn)
    assert recipe is not None and recipe.tile == 0


def test_sequential_loops_without_any_reduction_use_the_same_rule():
    second = POINTWISE[POINTWISE.index("for start"):].replace("src +", "dst +")
    recipe = match_local_axis(source_kernel(POINTWISE + second))
    assert recipe is not None
    assert len(recipe.accesses) == 4


@pytest.mark.parametrize("body", [POINTWISE, SUM, MAXIMUM, MINIMUM])
def test_tile_count_loops_follow_the_same_logical_axis(tensors, body):
    body = body.replace("range(0, length, width)", "range(tl.cdiv(length, width))")
    body = body.replace("index = start +", "index = start * width +")
    recipe = match_local_axis(source_kernel(body))
    assert recipe is not None
    args = (tensors(65536, count=201), tensors(131072, count=201), 67, 32)
    assert adapt_tile(recipe, args, {(3, ): 16}, (3, ), 0, None, compile_mode="simd") is not None


def test_loop_extent_cannot_depend_on_the_reused_tile():
    body = POINTWISE.replace("range(0, length, width)", "range(tl.cdiv(length + width, width))")
    body = body.replace("index = start +", "index = start * width +")
    assert match_local_axis(source_kernel(body)) is None


def test_broadcast_rows_and_columns_need_no_reduction(tensors):
    body = """
row = tl.program_id(0) * rows_per_program + tl.arange(0, rows_per_program)
row_mask = row < rows
for start in range(0, length, width):
    column = start + tl.arange(0, width)
    mask = row_mask[:, None] & (column[None, :] < length)
    address = row[:, None] * length + column[None, :]
    value = tl.load(src + address, mask=mask, other=0.0)
    tl.store(dst + address, tl.exp(value), mask=mask)
"""
    recipe = match_local_axis(
        source_kernel(body, parameters="src, dst, length, width, rows, rows_per_program", constexpr=(3, 5)))
    assert recipe is not None and recipe.tile == 3
    args = (tensors(65536, count=335), tensors(131072, count=335), 67, 32, 5, 2)
    assert adapt_tile(recipe, args, {(3, ): 16}, (3, ), 0, None, compile_mode="simd") is not None


@pytest.mark.parametrize("body", [
    POINTWISE.replace("* length", "* width"),
    POINTWISE.replace("result = tl.exp(value) + value * value", "result = value + width"),
    POINTWISE.replace("valid = index < length", "valid = index < width"),
    POINTWISE.replace("mask=valid", "mask=index < length - 1"),
    POINTWISE.replace("for start", "state = 1\nfor start").replace("    result =",
                                                                   "    state = state * 2\n    result ="),
    SUM.replace("other=0.0", "other=1.0"),
    SUM.replace("state += value * value", "state += value + 1.0"),
    SUM.replace("tl.sum(state, axis=0)", "tl.sum(state * state, axis=0)"),
    SUM.replace("tl.sum(state, axis=0)", "tl.max(state, axis=0)"),
    SUM.replace("tl.zeros((width,)", "tl.zeros((width, width)"),
    SUM.replace("state += value * value", "state += value * value\n    value = state + value"),
])
def test_tile_dependent_values_and_unproven_state_are_rejected(body):
    assert match_local_axis(source_kernel(body)) is None


def test_alias_check_is_based_on_accesses_not_parameter_roles(tensors):
    recipe = match_local_axis(source_kernel(SUM))
    tensor = tensors(65536, count=201)
    assert adapt_tile(recipe, (tensor, tensor, 67, 32), {(3, ): 16}, (3, ), 0, None, compile_mode="simd") is None


def test_runtime_extent_can_be_an_expression(tensors):
    body = POINTWISE.replace("length", "(length + 2)")
    recipe = match_local_axis(source_kernel(body))
    assert recipe is not None
    args = (tensors(65536, count=207), tensors(131072, count=207), 67, 32)
    assert adapt_tile(recipe, args, {(3, ): 16}, (3, ), 0, None, compile_mode="simd") is not None
