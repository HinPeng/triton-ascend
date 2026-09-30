import json
from pathlib import Path

import pytest
from test_analysis import Pointer, helper_entry, profile, scalar_store
from triton.backends.ascend.reuse.protocol import publish_manifest, read_manifest
from triton.backends.ascend.reuse.snapshot import SourceNotSerializable, bundle, decode, encode, rebuild
from triton.backends.ascend.reuse.transform import FrozenJITFunction, ReuseASTSource


@pytest.mark.parametrize("function", [helper_entry, scalar_store])
def test_round_trip_has_same_native_jit_identity(function):
    original = function.cache_key
    transported = json.loads(json.dumps(bundle(function)))
    restored = rebuild(transported)
    assert restored.cache_key == original
    assert restored.src == function.src
    assert restored.signature == function.signature


def test_transformed_source_and_abi_round_trip():
    clone = FrozenJITFunction.from_plan(profile(helper_entry, Pointer(), -1).plan)
    restored = rebuild(json.loads(json.dumps(bundle(clone))))
    assert restored.reuse_identity == clone.reuse_identity
    signature = {"out": "*i32", "PAD": "i32"}
    assert ReuseASTSource(restored, signature).hash() == ReuseASTSource(clone, signature).hash()
    assert not restored.params[1].is_constexpr
    assert restored.params[1].annotation_type == ""


def test_transitive_source_stamp_invalidates_on_helper_edit():
    from test_analysis import helper
    from triton.backends.ascend.reuse.dsl.source import source_stamp
    original = helper.src
    before = source_stamp(helper_entry)
    assert before.matches()
    try:
        helper._unsafe_update_src(original.replace("tl.store(out, PAD)", "tl.store(out, PAD + 1)"))
        assert not before.matches()
        after = source_stamp(helper_entry)
        assert after.matches() and after.token != before.token
    finally:
        helper._unsafe_update_src(original)


def test_snapshot_does_not_retain_unused_business_globals(monkeypatch):
    sentinel = object()
    monkeypatch.setitem(scalar_store.__globals__, "unused_business_tensor", sentinel)
    clone = FrozenJITFunction.from_plan(profile(scalar_store, Pointer(), 3, 1).plan)
    assert "unused_business_tensor" not in clone.get_capture_scope()


def test_business_objects_and_business_module_imports_are_rejected():
    with pytest.raises(SourceNotSerializable):
        encode(Pointer())
    with pytest.raises(SourceNotSerializable):
        decode(["module", "test_analysis"])


def test_constexpr_capture_encoding_preserves_triton_global_contract():
    import triton.language as tl
    value = decode(encode(tl.constexpr(3)))
    assert isinstance(value, tl.constexpr) and value.value == 3


def test_rule_version_changes_rewritten_identity_only(monkeypatch):
    from triton.backends.ascend.reuse import config
    from triton.compiler import ASTSource
    clone = FrozenJITFunction.from_plan(profile(scalar_store, Pointer(), 3, 7).plan)
    original = ASTSource(scalar_store, {'out': '*i32', 'N': 'constexpr', 'PAD': 'constexpr'}, {(1, ): 3, (2, ): 7})
    rewritten = ReuseASTSource(clone, {'out': '*i32', 'N': 'i32', 'PAD': 'i32'})
    old_original, old_rewritten = original.hash(), rewritten.hash()
    monkeypatch.setattr(config, 'RULE_VERSION', config.RULE_VERSION + 1)
    assert original.hash() == old_original
    assert rewritten.hash() != old_rewritten


def test_manifest_rejects_corruption_and_late_generation(tmp_path):
    from types import SimpleNamespace
    key = "0" * 64
    binary = tmp_path / "test.npubin"
    metadata = tmp_path / "test.json"
    binary.write_bytes(b"device binary")
    metadata.write_text(json.dumps({"hash": key}))
    request = {
        "schema": 1, "variant_key": key, "source": {}, "environment": {"cache": str(tmp_path)}, "generation": 4,
        "attempt": 0
    }
    kernel = SimpleNamespace(metadata_group={"test.npubin": str(binary), "test.json": str(metadata)})
    path = publish_manifest(request, kernel)
    assert read_manifest(path, request) == kernel.metadata_group
    late = dict(request, generation=5)
    with pytest.raises(ValueError, match="generation"):
        read_manifest(path, late)
    binary.write_bytes(b"broken")
    with pytest.raises(ValueError, match="digest"):
        read_manifest(path, request)


def test_parser_dependencies_stay_independent():
    from triton.backends.ascend import reuse
    root = Path(reuse.__file__).parent
    for path in root.rglob("*.py"):
        source = path.read_text()
        assert "runtime.dsl_analysis" not in source
        assert "runtime.autoparser" not in source
