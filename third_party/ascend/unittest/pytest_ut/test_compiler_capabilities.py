"""Optional flags work with both legacy and current NPUIR executables."""
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "_ascend_compiler_capabilities_test", Path(__file__).parents[2] / "backend/compiler_capabilities.py")
capabilities = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capabilities)


@pytest.fixture
def compiler(tmp_path):
    capabilities._supported_options.cache_clear()
    path = tmp_path / "bishengir-compile"
    path.write_bytes(b"compiler")
    return path


@pytest.mark.parametrize("supported,enabled,expected", [
    (False, False, []), (False, True, []),
    (True, False, ["--enable-optimized-metaop=False"]),
    (True, True, ["--enable-optimized-metaop=True"]),
])
def test_optional_metaop_flag(monkeypatch, compiler, supported, enabled, expected):
    help_text = "  --target=<arch> target\n"
    if supported:
        help_text += "  --enable-optimized-metaop=<bool> enable optimized templates\n"
    monkeypatch.setattr(capabilities.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=help_text))
    assert capabilities.optimized_metaop_options(compiler, enabled) == expected


def test_similar_option_is_not_support(monkeypatch, compiler):
    monkeypatch.setattr(capabilities.subprocess, "run", lambda *args, **kwargs:
        SimpleNamespace(returncode=0, stdout="  --enable-optimized-metaop-debug=<bool>\n"))
    assert capabilities.optimized_metaop_options(compiler, False) == []


@pytest.mark.parametrize("kind", ["timeout", "failed", "empty"])
def test_failed_probe_is_not_assumed_legacy(monkeypatch, compiler, kind):
    def probe(*args, **kwargs):
        if kind == "timeout":
            raise subprocess.TimeoutExpired(args[0], 15)
        return SimpleNamespace(returncode=1 if kind == "failed" else 0,
                               stdout="", stderr="probe failed")
    monkeypatch.setattr(capabilities.subprocess, "run", probe)
    with pytest.raises(RuntimeError, match="[Cc]apabilit|option information"):
        capabilities.optimized_metaop_options(compiler, False)


def test_probe_cache_refreshes_when_compiler_changes(monkeypatch, compiler):
    calls = []
    def probe(*args, **kwargs):
        calls.append(args)
        assert args[0][1] == "--help-hidden"
        return SimpleNamespace(returncode=0, stdout="  --target=<arch>\n" +
            ("  --enable-optimized-metaop\n" if len(calls)>1 else ""))
    monkeypatch.setattr(capabilities.subprocess, "run", probe)
    assert capabilities.optimized_metaop_options(compiler, False) == []
    assert capabilities.optimized_metaop_options(compiler, False) == []
    assert len(calls) == 1
    compiler.write_bytes(b"upgraded compiler")
    assert capabilities.optimized_metaop_options(compiler, False) == ["--enable-optimized-metaop=False"]
    assert len(calls) == 2
