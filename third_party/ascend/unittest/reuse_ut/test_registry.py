from types import SimpleNamespace

import pytest
from triton.backends.ascend.reuse import config
from triton.backends.ascend.reuse.context import InvocationFrame, exact_request_scope, requires_exact_request
from triton.backends.ascend.reuse.identity import typed
from triton.backends.ascend.reuse.registry import VariantRegistry


def objects(key="key", family=("family", )):
    src = SimpleNamespace(signature={"n": "i32"}, constants={}, attrs={}, hash=lambda: "source")
    kernel = SimpleNamespace(src=src, hash=key, module=1, function=2, _run=lambda: None, _initialization_complete=True)
    spec = SimpleNamespace(source=src, variant_key=key, family_key=family)
    return kernel, spec


def test_both_indexes_share_the_successful_object():
    registry = VariantRegistry()
    kernel, spec = objects()
    value = registry.register_ready_if_eligible(kernel, spec, (123, 0))
    assert registry.lookup_exact_ready("key", (123, 0)) is value
    assert registry.lookup_family_ready(("family", ), (123, 0)) == (value, )
    assert registry.register_ready_if_eligible(kernel, spec, (123, 0)) is value
    assert len(registry.exact) == 1
    assert registry.lookup_exact_ready("key", (124, 0)) is None


def test_new_proof_can_register_same_binary_without_new_object():
    registry = VariantRegistry()
    kernel, old = objects()
    value = registry.register_ready_if_eligible(kernel, old, (1, 0))
    _, current = objects(family=("new-proof", ))
    assert not registry.has_registration(value, current.family_key)
    assert registry.register_ready_if_eligible(kernel, current, (1, 0)) is value
    assert registry.has_registration(value, current.family_key)
    assert len(registry.exact) == 1


def test_late_success_cannot_reenable_an_open_circuit():
    registry = VariantRegistry()
    kernel, spec = objects()
    for _ in range(config.FAILURE_THRESHOLD):
        registry.record_transform_failure(spec.family_key, spec.variant_key)
    assert registry.register_ready_if_eligible(kernel, spec, (1, 0)) is None
    assert registry.disabled(spec.family_key, spec.variant_key)
    assert not registry.disabled(spec.family_key, "different-variant")


@pytest.mark.parametrize("field,value", [("_initialization_complete", False), ("module", None), ("function", 0),
                                         ("_run", None), ("hash", "wrong")])
def test_partial_or_wrong_artifacts_never_publish(field, value):
    registry = VariantRegistry()
    kernel, spec = objects()
    setattr(kernel, field, value)
    assert registry.register_ready_if_eligible(kernel, spec, (123, 0)) is None
    assert not registry.exact and not registry.families


def test_quota_skips_registration_only(monkeypatch):
    monkeypatch.setattr(config, "MAX_READY", 1)
    registry = VariantRegistry()
    kernel, spec = objects()
    registry.register_ready_if_eligible(kernel, spec, (1, 0))
    other, other_spec = objects("other")
    assert registry.register_ready_if_eligible(other, other_spec, (1, 0)) is None
    assert other._initialization_complete and callable(other._run)


def test_exact_scope_restores_after_error():
    assert not requires_exact_request()
    with pytest.raises(RuntimeError), exact_request_scope():
        assert requires_exact_request()
        with exact_request_scope():
            assert requires_exact_request()
        raise RuntimeError()
    assert not requires_exact_request()


def test_grid_frame_resolves_once_even_during_fallback():
    frame = InvocationFrame()
    calls = []

    def grid(meta):
        calls.append(meta)
        return (3, )

    assert frame.resolve_grid(grid, {"n": 3}) == (3, )
    assert frame.resolve_grid(grid, {"n": 4}) == (3, )
    assert len(calls) == 1


def test_typed_constants_do_not_alias():
    assert typed(True) != typed(1)
    assert typed(0.0) != typed(-0.0)
    assert typed(None) != typed("None")


def test_concurrent_initializers_share_one_loaded_object():
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    registry = VariantRegistry()
    first, spec = objects()
    second, _ = objects()
    count = []
    started = threading.Event()

    def initialize():
        count.append(1)
        started.set()
        time.sleep(0.01)

    first._init_handles = initialize
    second._init_handles = lambda: pytest.fail("duplicate native initialization")
    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(registry.initialize, first, spec, (1, 0))
        assert started.wait(timeout=1)
        other = pool.submit(registry.initialize, second, spec, (1, 0))
        assert future.result() is other.result() is first
    assert count == [1]


def test_unvalidated_background_starts_no_threads():
    from triton.backends.ascend.reuse.service import PreparationService
    registry = VariantRegistry()
    _kernel, spec = objects()
    service = PreparationService(registry, {})
    assert not service.try_submit(spec, (1, 0))
    assert service.compiler_executor is None and service.loader_executor is None and service.worker is None
    service.close()


def test_argument_error_precedes_option_parsing_without_creating_service(monkeypatch):
    import inspect

    from triton.backends.ascend.reuse import bridge
    monkeypatch.setattr(bridge, 'requires_original', lambda *args: False)

    def signature(a, b):
        pass

    fn = SimpleNamespace(signature=inspect.signature(signature))

    def forbidden(*args, **kwargs):
        pytest.fail('invalid binding reached options or JIT tail')

    backend = SimpleNamespace(parse_options=forbidden)
    handler = bridge.ReuseHandler()
    with pytest.raises(TypeError, match='missing'):
        handler.run(fn, (), {'compile_mode': 'invalid'}, (1, ), False, 0, 0, backend, forbidden)
    assert handler.registry is None
