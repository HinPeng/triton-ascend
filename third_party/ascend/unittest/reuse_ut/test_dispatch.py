from types import SimpleNamespace

from test_registry import objects
from triton.backends.ascend.reuse.dispatch import resolve_dispatch
from triton.backends.ascend.reuse.model import JitTailRequest, PreparedLaunch
from triton.backends.ascend.reuse.registry import VariantRegistry


def test_exact_ready_does_not_scan_family(monkeypatch):
    registry = VariantRegistry()
    kernel, spec = objects()
    registry.register_ready_if_eligible(kernel, spec, (1, 0))

    def forbidden(*args):
        raise AssertionError("exact hit scanned the family")

    monkeypatch.setattr(registry, 'lookup_family_ready', forbidden)
    result = resolve_dispatch(registry, spec, None, (7, ), (1, ), (1, 0), None)
    assert isinstance(result, PreparedLaunch)
    assert result.executable.kernel is kernel
    assert result.arguments == (7, )


def test_two_layer_miss_only_requests_original_jit_tail():
    registry = VariantRegistry()
    _, spec = objects()
    result = resolve_dispatch(registry, spec, None, (), (1, ), (1, 0), None)
    assert isinstance(result, JitTailRequest) and result.desired is spec
    assert registry.snapshot() == {'exact_unavailable': 1, 'family_exhausted': 1, 'jit_tail_miss': 1}


def test_rejected_family_candidate_does_not_hide_a_later_one(monkeypatch):
    from triton.backends.ascend.reuse import dispatch
    registry = VariantRegistry()
    first, first_spec = objects('first')
    second, second_spec = objects('second')
    first.src.constants = {(0, ): 16}
    second.src.constants = {(0, ): 32}
    registry.register_ready_if_eligible(first, first_spec, (1, 0))
    registry.register_ready_if_eligible(second, second_spec, (1, 0))
    _, desired = objects('desired')
    desired.options = SimpleNamespace(compile_mode='simd_simt_template')
    second.src.signature = {'n': 'constexpr'}
    binder = lambda *args, **kwargs: ({'n': args[0]}, [('constexpr', args[0])], {})
    desired.owner = SimpleNamespace(device_caches={0: (None, None, None, None, binder)}, arg_names=['n'])
    attempted = []

    def adapt(recipe, args, constants, grid, device, *, compile_mode):
        assert compile_mode == 'simd_simt_template'
        attempted.append(constants[(0, )])
        return None if constants[(0, )] == 16 else (32, )

    monkeypatch.setattr(dispatch, 'adapt_tile', adapt)
    result = resolve_dispatch(registry, desired, SimpleNamespace(recipe=object()), (64, ), (1, ), (1, 0), None)
    assert isinstance(result, PreparedLaunch) and result.executable.kernel is second
    assert attempted == [16, 32]
