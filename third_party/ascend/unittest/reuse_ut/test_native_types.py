"""The rewritten ABI delegates scalar types and type-dependent keys to JIT."""
import ast
import inspect
from types import SimpleNamespace

import pytest
import triton
import triton.language as tl
from test_analysis import Pointer, profile, scalar_store
from triton.backends.ascend.reuse.transform import FrozenJITFunction
from triton.runtime import jit


@triton.jit
def direct_store(out, value: tl.constexpr):
    tl.store(out, value)


@triton.jit
def numeric_operations(out, N: tl.constexpr, DIV: tl.constexpr):
    value: tl.constexpr = (N << 2) / DIV
    tl.store(out, tl.cast(tl.cdiv(value, DIV), tl.float32))


@pytest.mark.parametrize("n,div", [(2**40, 3), (-7, -2), (3, 0), (0.5, 0.25)])
def test_numeric_operations_reach_native_compilation_without_preflight(n, div):
    p = profile(numeric_operations, Pointer("float32"), n, div)
    assert p.plan.dynamic == (1, 2)
    clone = FrozenJITFunction.from_plan(p.plan)
    assert all(not parameter.is_constexpr for parameter in clone.params)


@triton.jit
def floating_shape(out, N: tl.constexpr):
    offsets = tl.arange(0, N)
    tl.store(out + offsets, 0)


def test_scalar_support_does_not_remove_shape_requirements():
    from test_analysis import decision
    from triton.backends.ascend.reuse.model import ParameterClass
    p = profile(floating_shape, Pointer(), 16.0)
    assert not p.plan.dynamic
    assert decision(p, 1).classification is ParameterClass.Unknown
    assert ("TILE_REUSE_UNPROVEN", 0) in decision(p, 1).reasons


@pytest.mark.parametrize(
    "value", [3, 1, 2**31, 2**40, -(2**63), 2**64 - 1, 0.5, True,
              float("nan"), float("inf"), -0.0])
def test_direct_scalar_transport_has_no_dtype_or_range_gate(value):
    p = profile(direct_store, Pointer("float32"), value)
    assert p.plan.dynamic == (1, )
    clone = FrozenJITFunction.from_plan(p.plan)
    assert clone.params[1].annotation_type == ""


def test_arithmetic_does_not_impose_a_fixed_integer_domain():
    p = profile(scalar_store, Pointer(), 2**40, 2**63)
    assert p.plan.dynamic == (1, 2)


def test_numeric_subclasses_share_scalar_analysis_context():
    from triton.backends.ascend.reuse.analysis import type_context

    class Integer(int):
        pass

    class Floating(float):
        pass

    for value in (Integer(2**40), Floating(0.5)):
        p = profile(direct_store, Pointer(), value)
        assert p.plan.dynamic == (1, )
        assert type_context(direct_store, (Pointer(), value))[1][0] in ("int", "float")


@pytest.mark.parametrize("value,native_type", [(1, "i32"), (2**40, "i64"), (2**63, "u64"), (0.5, "fp32"), (True, "u1")])
def test_unannotated_binder_uses_native_type_without_overriding_it(monkeypatch, value, native_type):
    clone = FrozenJITFunction.from_plan(profile(direct_store, Pointer("int64"), value).plan)
    assert clone.signature.parameters["value"].annotation is inspect.Parameter.empty
    assert clone.params[1].annotation_type == "" and not clone.params[1].is_constexpr
    assert ast.parse(clone.src).body[0].args.args[1].annotation is None
    backend = SimpleNamespace(use_alignment_specialization=lambda options: True)
    output = Pointer("int64")
    observed = []

    def native(actual_backend, argument, is_const, specialize_value, align):
        assert actual_backend is backend
        if argument is output:
            return "*i64", None
        observed.append((argument, is_const, specialize_value, align))
        return native_type, None

    monkeypatch.setattr(jit, "native_specialize_impl", native)
    binder = jit.create_function_from_signature(clone.signature, clone.params, backend)
    bound, specialization, _ = binder(output, value)
    assert bound["value"] == value
    assert specialization[1] == (native_type, None)
    assert observed == [(value, False, False, False)]
    assert direct_store.params[1].is_constexpr


def test_native_signature_changes_keep_distinct_cache_keys(monkeypatch):
    clone = FrozenJITFunction.from_plan(profile(direct_store, Pointer("int64"), 3).plan)
    output = Pointer("int64")
    backend = SimpleNamespace(use_alignment_specialization=lambda options: True)
    # Controlled native results test the delegation/key contract. Actual inference
    # and compilation across the i32/i64 boundary are covered by the NPU test.
    native_results = {3: "i32", 4: "i32", 2**40: "i64", 2**41: "i64"}

    def native(actual_backend, argument, is_const, specialize_value, align):
        return ("*i64", None) if argument is output else (native_results[argument], None)

    monkeypatch.setattr(jit, "native_specialize_impl", native)
    binder = jit.create_function_from_signature(clone.signature, clone.params, backend)
    keys = []
    cache = {}
    for value in native_results:
        _, specialization, options = binder(output, value)
        keys.append(jit.compute_cache_key(cache, specialization, options))
    assert keys[0] == keys[1]
    assert keys[2] == keys[3]
    assert keys[0] != keys[2]
