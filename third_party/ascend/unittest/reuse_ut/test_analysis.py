import inspect

import pytest
import triton
import triton.language as tl
from triton.backends.ascend.reuse.analysis import analyze
from triton.backends.ascend.reuse.binding import ArgumentBinder
from triton.backends.ascend.reuse.model import ParameterClass
from triton.backends.ascend.reuse.transform import FrozenJITFunction


@triton.jit
def scalar_store(out, N: tl.constexpr, PAD: tl.constexpr):
    extent: tl.constexpr = N * 2 + 1
    i = tl.program_id(0)
    if i < extent:
        tl.store(out + i, PAD)


@triton.jit
def overwritten(out, N: tl.constexpr):
    limit = N
    offsets = tl.arange(0, limit)
    limit = 32
    tl.store(out + offsets, limit)


@triton.jit
def static_branch(out, N: tl.constexpr):
    if N == 1:
        tl.store(out, 1)
    else:
        tl.store(out, N)


@triton.jit
def static_loop(out, COUNT: tl.constexpr, SCALE: tl.constexpr):
    for i in tl.static_range(COUNT):
        tl.store(out + i, SCALE)


@triton.jit
def helper(out, PAD: tl.constexpr = 3):
    tl.store(out, PAD)


@triton.jit
def helper_entry(out, PAD: tl.constexpr):
    helper(out, PAD=PAD)
    helper(out + 1, PAD)


@triton.jit
def unknown(out, N: tl.constexpr):
    tl.histogram(tl.arange(0, 32), N)


@triton.jit
def augmented(out, N: tl.constexpr):
    size = 1
    for i in range(2):
        size += N
    offsets = tl.arange(0, size)
    tl.store(out + offsets, 0)


@triton.jit
def floating(out, SCALE: tl.constexpr):
    x = tl.load(out)
    tl.store(out, x * SCALE)


@triton.jit
def one_specialized(out, runtime_n, N: tl.constexpr):
    if runtime_n + N > 0:
        tl.store(out, N)


@triton.jit
def positional_load_flag(ptr, out, VOLATILE: tl.constexpr):
    value = tl.load(ptr, True, 0, (), "", "", "", VOLATILE)
    tl.store(out, value)


@triton.jit
def keyword_where(out, N: tl.constexpr):
    value = tl.where(y=tl.load(out), x=N, condition=True)
    tl.store(out, value)


class Pointer:

    def __init__(self, dtype="int32"):
        self.dtype = dtype

    def data_ptr(self):
        raise AssertionError("analysis must not read addresses")


def profile(fn, *args):
    bound, _ = ArgumentBinder(fn.signature)(*args)
    return analyze(fn, bound, fn.cache_key)


def decision(p, position):
    return next(d for d in p.decisions if d.position == position)


def test_dynamic_and_local_constexpr_snapshot():
    original = (scalar_store.src, scalar_store.signature, scalar_store.params, scalar_store.device_caches)
    p = profile(scalar_store, Pointer(), 3, -1)
    assert p.plan.dynamic == (1, 2)
    clone = FrozenJITFunction.from_plan(p.plan)
    assert not clone.params[1].is_constexpr and clone.params[1].annotation_type == ""
    assert clone.signature.parameters["N"].annotation is inspect.Parameter.empty
    assert clone.params[1].do_not_specialize and clone.params[1].do_not_specialize_on_alignment
    assert "extent: " not in clone.src
    assert original == (scalar_store.src, scalar_store.signature, scalar_store.params, scalar_store.device_caches)
    assert clone.device_caches is not scalar_store.device_caches


@pytest.mark.parametrize("fn", [overwritten, static_branch, augmented])
def test_static_uses_survive_overwrite_and_control(fn):
    p = profile(fn, Pointer(), 1)
    assert not p.plan.dynamic
    assert decision(p, 1).classification is ParameterClass.StaticRequired


def test_static_loop_body_is_analyzed_independently():
    p = profile(static_loop, Pointer(), 4, 7)
    assert p.plan.dynamic == (2, )
    assert decision(p, 1).classification is ParameterClass.StaticRequired


def test_all_helper_calls_are_transformed_without_mutation():
    src = helper.src
    p = profile(helper_entry, Pointer(), -7)
    assert p.plan.dynamic == (1, )
    assert len(p.plan.helpers) == 2
    clone = FrozenJITFunction.from_plan(p.plan)
    copies = [v for k, v in clone.get_capture_scope().items() if k.startswith("_reuse_helper_")]
    assert len(copies) == 2 and all(not f.params[1].is_constexpr for f in copies)
    assert helper.src == src and helper.params[1].is_constexpr
    assert clone.cache_key == FrozenJITFunction.from_plan(p.plan).cache_key


def test_unknown_operation_is_not_a_positive_proof():
    p = profile(unknown, Pointer(), 32)
    assert not p.plan.dynamic
    assert decision(p, 1).classification is ParameterClass.Unknown
    assert "OPERATION_SUMMARY_MISSING" in repr(decision(p, 1).reasons)


def test_integer_to_float_arithmetic_is_delegated_to_triton():
    p = profile(floating, Pointer("float16"), 2)
    assert p.plan.dynamic == (1, )


@pytest.mark.parametrize("dtype", ["int8", "int16", "uint32", "uint64"])
def test_narrow_or_unsigned_integer_promotion_is_delegated_to_triton(dtype):
    p = profile(floating, Pointer(dtype), 1000)
    assert p.plan.dynamic == (1, )


def test_native_one_specialization_is_part_of_the_control_type_context():
    from triton.backends.ascend.reuse.analysis import type_context
    assert not profile(one_specialized, Pointer(), 1, 3).plan.dynamic
    assert profile(one_specialized, Pointer(), 2, 3).plan.dynamic == (2, )
    assert type_context(one_specialized, (Pointer(), 1, 3)) != type_context(one_specialized, (Pointer(), 2, 3))


@pytest.mark.parametrize("n,pad", [(1, 1), (3, -(2**31)), (3, 2**31 - 1), (2**30, 1),
                                   (2**40, 2**63), (3, True), (3, 0.5)])
def test_scalar_analysis_does_not_precheck_numeric_domains(n, pad):
    p = profile(scalar_store, Pointer(), n, pad)
    assert p.plan.dynamic == (1, 2)


def test_boolean_control_stays_static_while_boolean_data_is_eligible():
    p = profile(static_branch, Pointer(), True)
    assert decision(p, 1).classification is ParameterClass.StaticRequired


def test_positional_instruction_flags_are_static():
    p = profile(positional_load_flag, Pointer(), Pointer(), 1)
    assert decision(p, 2).classification is ParameterClass.StaticRequired


def test_where_keyword_order_keeps_scalar_data_dependencies():
    p = profile(keyword_where, Pointer('float16'), 3)
    assert p.plan.dynamic == (1, )


def test_argument_binder_defaults_options_and_errors():

    def f(x, size=4):
        pass

    bind = ArgumentBinder(inspect.signature(f))
    assert bind(7, num_warps=8) == ({"x": 7, "size": 4}, {"num_warps": 8})
    with pytest.raises(TypeError, match="multiple values"):
        bind(7, x=8)
    with pytest.raises(TypeError, match="missing"):
        bind()
