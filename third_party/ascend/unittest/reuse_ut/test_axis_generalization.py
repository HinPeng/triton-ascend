"""Equivalent index spellings, axis provenance, static paths and JIT helpers."""
import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import triton.language as tl
from test_local_axis import POINTWISE, source_kernel
from test_tile_rules import tensors as tensor_fixture
from triton.backends.ascend.reuse.analysis import analyze, static_context
from triton.backends.ascend.reuse.guards import adapt_tile
from triton.backends.ascend.reuse.local_axis import match_local_axis
from triton.runtime.jit import JITFunction

tensors = tensor_fixture


def adapted(recipe, tensors, *extra):
    args = (tensors(65536, count=201), tensors(131072, count=201), 67, 32, *extra)
    return adapt_tile(recipe, args, {(3, ): 16}, (3, ), 0, compile_mode="simd")


@pytest.mark.parametrize("index", [
    "tl.arange(0, width) + start",
    "start + (tl.arange(0, width) + 0)",
    "(start + 5) + (tl.arange(0, width) - 5)",
    "(start * 1) + tl.arange(0, 1 * width)",
    "start + tl.arange(0, width) - 0",
])
def test_associative_index_spellings(tensors, index):
    body = POINTWISE.replace("start + tl.arange(0, width)", index)
    recipe = match_local_axis(source_kernel(body))
    assert recipe is not None and adapted(recipe, tensors) is not None


def test_reversed_comparison_and_pointer_addition(tensors):
    body = POINTWISE.replace("index < length", "length > index")
    body = body.replace("src + base + index", "index + (base + src)")
    recipe = match_local_axis(source_kernel(body))
    assert recipe is not None and adapted(recipe, tensors) is not None


@pytest.mark.parametrize("count", ["(length + width - 1) // width", "(width + (length - 1)) // width"])
def test_integer_ceil_division_spellings(tensors, count):
    body = POINTWISE.replace("range(0, length, width)", f"range({count})")
    body = body.replace("index = start +", "index = (1 * width) * start +")
    recipe = match_local_axis(source_kernel(body))
    assert recipe is not None and recipe.loops[0].counts_tiles
    assert adapted(recipe, tensors) is not None


def test_original_overflow_is_not_hidden_by_constant_cancellation(tensors):
    body = POINTWISE.replace("start + tl.arange(0, width)", "(start + 2147483647) + (tl.arange(0, width) - 2147483647)")
    recipe = match_local_axis(source_kernel(body))
    assert recipe is not None
    assert adapted(recipe, tensors) is None


@pytest.mark.parametrize("annotation", ["i8", "u32"])
def test_explicit_narrow_indices_are_not_reassociated_as_int32(annotation):
    fn = source_kernel(POINTWISE.replace("length", "(length + 1)"))
    fn.params[2].annotation_type = annotation
    assert match_local_axis(fn) is None


def test_floating_division_is_not_integer_ceil_division():
    body = POINTWISE.replace("range(0, length, width)", "range((length + width - 1) / width)")
    assert match_local_axis(source_kernel(body)) is None


def test_float_cast_indices_are_not_normalized_as_integer_indices():
    body = POINTWISE.replace("start + tl.arange(0, width)", "start + tl.arange(0, width).to(tl.float32)")
    assert match_local_axis(source_kernel(body)) is None


def test_distinct_loops_share_axis_identity_but_keep_provenance():
    second = POINTWISE[POINTWISE.index("for start"):].replace("src +", "dst +")
    recipe = match_local_axis(source_kernel(POINTWISE + second))
    assert recipe is not None and len(recipe.axes) == 1 and len(recipe.loops) == 2
    assert recipe.loops[0].axis == recipe.loops[1].axis == recipe.axes[0]
    assert recipe.loops[0].site != recipe.loops[1].site
    assert recipe.loops[0].line != recipe.loops[1].line


def test_stale_loop_coordinates_are_not_rebound_to_a_later_loop():
    body = POINTWISE + POINTWISE[POINTWISE.index("for start"):].replace("    index = start + tl.arange(0, width)\n", "")
    assert match_local_axis(source_kernel(body)) is None


def branching_kernel():
    body = "if MODE:\n    chunk = width\nelse:\n    chunk = width + 1\n" + POINTWISE.replace(
        "tl.arange(0, width)", "tl.arange(0, chunk)")
    return source_kernel(body, parameters="src, dst, length, width, MODE", constexpr=(3, 4))


def test_static_branch_selection_and_cache_context(tensors):
    fn = branching_kernel()
    good = match_local_axis(fn, {"MODE": True})
    assert good is not None and good.static_parameters == (4, )
    assert adapted(good, tensors, True) is not None
    assert match_local_axis(fn, {"MODE": False}) is None
    assert static_context(fn, (None, None, 67, 32, True)) != static_context(fn, (None, None, 67, 32, False))


def test_analysis_retains_control_parameters():
    fn = branching_kernel()
    for i, (name, parameter) in enumerate(zip(fn.arg_names, fn.params)):
        parameter.num, parameter.name = i, name
        parameter.annotation_type, parameter.do_not_specialize = "", False
    arguments = dict(zip(fn.arg_names, (None, None, 67, 32, True)))
    profile = analyze(fn, arguments, "static-control")
    assert profile.recipe is not None and profile.recipe.static_parameters == (4, )
    assert 4 not in profile.plan.dynamic
    arguments["MODE"] = False
    assert analyze(fn, arguments, "static-control").recipe is None


def test_static_float_selector_cache_uses_typed_values():
    fn = branching_kernel()
    assert match_local_axis(fn, {"MODE": 0.5}) is not None
    assert match_local_axis(fn, {"MODE": 0.0}) is None
    assert static_context(fn, (None, None, 67, 32, 0.0)) != static_context(fn, (None, None, 67, 32, -0.0))


def test_conjunction_order_and_identity(tensors):
    body = POINTWISE.replace("valid = index < length", "valid = True & (length > index)")
    recipe = match_local_axis(source_kernel(body))
    assert recipe is not None and adapted(recipe, tensors) is not None


def test_runtime_and_tile_dependent_branches_decline():
    fn = branching_kernel()
    fn.params[4].is_constexpr = False
    assert match_local_axis(fn, {"MODE": True}) is None
    body = "if width > 0:\n    chunk = width\nelse:\n    chunk = width\n" + POINTWISE
    assert match_local_axis(source_kernel(body), {"width": 32}) is None


def test_static_if_expression_and_annotated_alias(tensors):
    body = "chunk: tl.constexpr = width if MODE == 2 else width + 1\n" + POINTWISE.replace(
        "tl.arange(0, width)", "tl.arange(0, chunk)")
    fn = source_kernel(body, parameters="src, dst, length, width, MODE", constexpr=(3, 4))
    recipe = match_local_axis(fn, {"MODE": 2})
    assert recipe is not None and recipe.static_parameters == (4, )
    assert adapted(recipe, tensors, 2) is not None


def helper(source, constexpr=(), scope=None):
    """JITFunction metadata fixture; native compilation is covered by NPU tests."""
    tree = ast.parse(textwrap.dedent(source))
    function = tree.body[0]
    defaults = [inspect.Parameter.empty] * (len(function.args.args) - len(function.args.defaults)) + [
        ast.literal_eval(n) for n in function.args.defaults
    ]
    fn = object.__new__(JITFunction)
    fn.__name__ = function.name
    fn.arg_names = [arg.arg for arg in function.args.args]
    fn.params = [SimpleNamespace(name=name, is_constexpr=name in constexpr) for name in fn.arg_names]
    fn.signature = inspect.Signature([
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default)
        for name, default in zip(fn.arg_names, defaults)
    ])
    fn.parse = lambda: ast.parse(textwrap.dedent(source))
    fn.get_capture_scope = lambda: {"tl": tl, **(scope or {})}
    return fn


def with_helpers(fn, **helpers):
    fn.get_capture_scope = lambda: {"tl": tl, **helpers}
    return fn


def test_helper_index_mask_and_tuple_return(tensors):
    make = helper(
        """
def make(start, size, extent):
    index = tl.arange(0, size) + (start + 0)
    return index, extent > index
""", constexpr=("size", ))
    body = POINTWISE.replace("index = start + tl.arange(0, width)\n    valid = index < length",
                             "index, valid = make(size=width, extent=length, start=start)")
    recipe = match_local_axis(with_helpers(source_kernel(body), make=make))
    assert recipe is not None and adapted(recipe, tensors) is not None


def test_nested_helpers_preserve_load_semantics_and_defaults(tensors):
    load = helper(
        """
def read(pointer, offset, mask, pad=0):
    return tl.load(pointer + offset, mask=mask, other=pad)
""", constexpr=("pad", ))
    wrapper = helper(
        """
def fetch(pointer, offset, mask):
    value = read(pointer, offset, mask=mask)
    return value * value
""", scope={"read": load})
    body = POINTWISE.replace("tl.load(src + base + index, mask=valid, other=0.0)", "fetch(src, base + index, valid)")
    recipe = match_local_axis(with_helpers(source_kernel(body), fetch=wrapper))
    assert recipe is not None and adapted(recipe, tensors) is not None


def test_static_helper_branch_retains_root_selector(tensors):
    make = helper(
        """
def make(start, size, mode=True):
    if mode:
        return start + tl.arange(0, size)
    return start + tl.arange(0, size + 1)
""", constexpr=("size", "mode"))
    body = POINTWISE.replace("start + tl.arange(0, width)", "make(start, width, mode=MODE)")
    fn = with_helpers(source_kernel(body, parameters="src, dst, length, width, MODE", constexpr=(3, 4)), make=make)
    recipe = match_local_axis(fn, {"MODE": True})
    assert recipe is not None and recipe.static_parameters == (4, )
    assert adapted(recipe, tensors, True) is not None
    assert match_local_axis(fn, {"MODE": False}) is None


def test_complete_traversal_can_live_in_a_helper(tensors):
    walk = helper("def walk(src, dst, length, width):\n" + textwrap.indent(POINTWISE.strip(), "    "),
                  constexpr=("width", ))
    recipe = match_local_axis(with_helpers(source_kernel("walk(src, dst, length, width)"), walk=walk))
    assert recipe is not None and recipe.loops[0].function == "walk"
    assert adapted(recipe, tensors) is not None


def test_early_return_from_traversal_is_not_full_coverage():
    walk = helper(
        "def walk(src, dst, length, width):\n" + textwrap.indent(POINTWISE.strip(), "    ") + "\n        return\n",
        constexpr=("width", ))
    assert match_local_axis(with_helpers(source_kernel("walk(src, dst, length, width)"), walk=walk)) is None


def test_recursive_and_non_jit_helpers_decline():
    recursive = helper("def recur(start, size):\n    return recur(start, size)\n", constexpr=("size", ))
    recursive.get_capture_scope = lambda: {"recur": recursive}
    body = POINTWISE.replace("start + tl.arange(0, width)", "recur(start, width)")
    assert match_local_axis(with_helpers(source_kernel(body), recur=recursive)) is None
    assert match_local_axis(with_helpers(source_kernel(body), recur=lambda *args: None)) is None
