"""Regression cases for call-site, arithmetic, type and tile reuse proofs."""
from types import SimpleNamespace

import pytest
import triton
import triton.language as tl

from test_analysis import Pointer, decision, profile
from triton.backends.ascend.reuse.analysis import type_context
from triton.backends.ascend.reuse.model import ParameterClass
from triton.backends.ascend.reuse.rules import match_grid_stride
from triton.backends.ascend.reuse.transform import FrozenJITFunction


@triton.jit
def add_one(N: tl.constexpr):
    return N + 1


@triton.jit
def add_two(N: tl.constexpr):
    return N + 2


@triton.jit
def subtract_one(M: tl.constexpr):
    return M - 1


@triton.jit
def same_line_helpers(out, N: tl.constexpr):
    value = add_one(N) + add_two(N)
    tl.store(out, value)


@triton.jit
def different_helper_parameters(out, N: tl.constexpr):
    value = add_one(N) + subtract_one(N)
    tl.store(out, value)


@triton.jit
def nested_helper_store(out, N: tl.constexpr):
    tl.store(out, add_one(add_two(N)))


def scalar_python_copy(fn):
    """Evaluate rewritten helper routing with Python integers, without a device."""
    scope = {
        name: scalar_python_copy(value) if isinstance(value, triton.JITFunction) else value
        for name, value in fn.get_capture_scope().items()
    }
    scope["tl"] = SimpleNamespace(store=lambda out, value: out.append(value))
    exec(fn.src, scope)  # noqa: S102 - execute only these arithmetic test fixtures
    return scope[fn.__name__]


@pytest.mark.parametrize("fn,expected", [(same_line_helpers, 9), (different_helper_parameters, 6),
                                         (nested_helper_store, 6)])
def test_helper_calls_on_one_line_keep_their_targets(fn, expected):
    original = fn.src
    p = profile(fn, Pointer(), 3)
    assert p.plan.dynamic == (1, )
    clone = FrozenJITFunction.from_plan(p.plan)
    out = []
    scalar_python_copy(clone)(out, 3)
    assert out == [expected]
    assert fn.src == original


@triton.jit
def inverted_division(out, N: tl.constexpr):
    tl.store(out, (~N) // 2)


@triton.jit
def inverted_remainder(out, N: tl.constexpr):
    tl.store(out, (~N) % 2)


@triton.jit
def positive_product(out, N: tl.constexpr):
    tl.store(out, (+N) * 65536)


@triton.jit
def inverted_product(out, N: tl.constexpr):
    tl.store(out, (~N) * 65536)


@triton.jit
def branch_expression(out, N: tl.constexpr, MODE: tl.constexpr):
    if MODE:
        value = N
    else:
        value = 0
    tl.store(out, value * 65536)


@pytest.mark.parametrize("fn", [inverted_division, inverted_remainder, positive_product, inverted_product])
@pytest.mark.parametrize("n", [4, 0, -5, -(2**31), 2**31, 2**40, 65536])
def test_integer_arithmetic_is_not_blocked_by_value_range_or_sign(fn, n):
    p = profile(fn, Pointer(), n)
    assert p.plan.dynamic == (1, )


def test_branch_merge_does_not_require_host_arithmetic_evaluation():
    p = profile(branch_expression, Pointer(), 65536, True)
    assert p.plan.dynamic == (1, )
    assert decision(p, 2).classification is ParameterClass.StaticRequired


@triton.jit
def pointwise(a, out, length, work, tile: tl.constexpr, scale):
    p = tl.program_id(0)
    step = tl.num_programs(0)
    for block in range(p, work, step):
        index = block * tile + tl.arange(0, tile)
        mask = index < length
        x = tl.load(a + index, mask=mask)
        result = x * scale
        tl.store(out + index, result, mask=mask)


def pointwise_copy(old, new):
    fn = triton.jit(pointwise.fn)
    assert old in pointwise.src
    fn._unsafe_update_src(pointwise.src.replace(old, new))
    return fn


@pytest.mark.parametrize("old,new", [
    ("result = x * scale", "result = x + work"),
    ("result = x * scale", "bias = work + 1\n        result = x + bias"),
    ("index < length", "index < work"),
    ("index < length", "index < tile"),
])
def test_tile_rule_rejects_semantic_uses_of_adapted_arguments(old, new):
    fn = pointwise_copy(old, new)
    assert match_grid_stride(fn) is None
    p = profile(fn, Pointer("float32"), Pointer("float32"), 1024, 2, 512, 0.5)
    assert decision(p, 4).classification is not ParameterClass.ScheduleReusable


@pytest.mark.parametrize("old,new", [
    ("result = x * scale", "scale = scale + 1\n        result = x * scale"),
    ("result = x * scale", "result = x * scale\n        scale = scale + 1"),
    ("result = x * scale", "previous = scale\n        scale = previous + 1\n        result = x * scale"),
])
def test_tile_rule_rejects_loop_carried_state(old, new):
    assert match_grid_stride(pointwise_copy(old, new)) is None


def test_tile_rule_keeps_independent_iterations():
    assert match_grid_stride(pointwise) is not None
    reset = pointwise_copy("result = x * scale", "scale = 0.5\n        result = x * scale\n        scale = scale + 1")
    assert match_grid_stride(reset) is not None


def scalar_annotation_kernel(annotation, do_not_specialize=False):

    def kernel(out, x, N: tl.constexpr):
        tl.store(out, ~(x + N))

    kernel.__annotations__["x"] = annotation
    return triton.jit(kernel, do_not_specialize=["x"] if do_not_specialize else [])


@pytest.mark.parametrize("annotation", ["i8", "i16", "u8", "u16", "u32", "u64", tl.uint16])
def test_explicit_narrow_and_unsigned_scalar_types_keep_native_promotion(annotation):
    fn = scalar_annotation_kernel(annotation)
    p = profile(fn, Pointer(), 5, 3)
    assert p.plan.dynamic == (2, )


@pytest.mark.parametrize("annotation", ["i32", "i64", tl.int32])
def test_supported_explicit_scalar_types_remain_dynamic(annotation):
    assert profile(scalar_annotation_kernel(annotation), Pointer(), 5, 3).plan.dynamic == (2, )


def test_explicit_type_and_native_one_specialization_are_both_preserved():
    fn = scalar_annotation_kernel("u16")
    assert type_context(fn, (Pointer(), 1, 3))[1] == ("int", True)
    assert profile(fn, Pointer(), 1, 3).plan.dynamic == (2, )
    unspecialized = scalar_annotation_kernel("u16", do_not_specialize=True)
    assert type_context(unspecialized, (Pointer(), 1, 3))[1] == ("uint16", False)
    assert profile(unspecialized, Pointer(), 1, 3).plan.dynamic == (2, )


@pytest.mark.parametrize("annotation,kind", [("fp32", "fp32"), ("u1", "bool")])
def test_float_and_boolean_annotations_do_not_specialize_integer_one(annotation, kind):
    fn = scalar_annotation_kernel(annotation)
    assert type_context(fn, (Pointer(), 1, 3))[1] == (kind, False)


def test_unannotated_integer_width_is_left_to_native_inference():
    fn = scalar_annotation_kernel("")
    assert type_context(fn, (Pointer(), 2**63, 3))[1] == ("int", False)
    assert profile(fn, Pointer(), 2**63, 3).plan.dynamic == (2, )


@triton.jit
def float_default_helper(out, N: tl.constexpr, SCALE: tl.constexpr = 0.1):
    tl.store(out, N * SCALE)


@triton.jit
def float_default_entry(out, N: tl.constexpr):
    float_default_helper(out, N)


@triton.jit
def overridden_default_entry(out, N: tl.constexpr):
    float_default_helper(out, N, SCALE=2)


@triton.jit
def integer_default_helper(out, N: tl.constexpr, SHIFT: tl.constexpr = 3):
    tl.store(out, N + SHIFT)


@triton.jit
def integer_default_entry(out, N: tl.constexpr):
    integer_default_helper(out, N)


def test_float_helper_default_keeps_dependency_rewriting():
    p = profile(float_default_entry, Pointer(), 16777219)
    assert p.plan.dynamic == (1, )
    assert p.plan.helpers[0][1].dynamic == (1, )


@pytest.mark.parametrize("fn", [integer_default_entry, overridden_default_entry])
def test_integer_helper_defaults_and_overrides_keep_large_values_dynamic(fn):
    p = profile(fn, Pointer(), 2**40)
    assert p.plan.dynamic == (1, )
    assert p.plan.helpers[0][1].dynamic == (1, )
