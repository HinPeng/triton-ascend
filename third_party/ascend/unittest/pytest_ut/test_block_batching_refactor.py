# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Compatibility contracts for arithmetic reuse and lazy launch policy."""
from dataclasses import FrozenInstanceError
import importlib
import random

import pytest
from test_block_batching_guard import batching, bundle_environment, guard_module, PACKAGE

pytestmark = pytest.mark.backend('native')


@pytest.mark.parametrize('rank', [0, 1, 4, 8, 19])
def test_affine_arithmetic_preserves_concrete_types_and_dimensions(rank):
    nd = importlib.import_module(PACKAGE + '.block_batching_guard_v2').AffineND
    rng = random.Random(9741 + rank)
    for cls in [guard_module.Affine, nd]:
        for _ in range(100):
            a, b = (cls(rng.randrange(-2**63, 2**63), tuple(rng.randrange(-8, 9)
                                                            for _ in range(rank)))
                    for _ in range(2))
            for sign in [-1, 1]:
                c = a.add(b, sign=sign)
                assert type(c) is cls
                assert c.offset == a.offset + sign * b.offset
                assert c.coefficients == tuple(x + sign * y for x, y in zip(a.coefficients, b.coefficients))
            scaled = a.scale(-3)
            assert type(scaled) is cls and scaled.offset == -3 * a.offset
            assert scaled.coefficients == tuple(-3 * x for x in a.coefficients)
            assert a.uniform == all(x == 0 for x in a.coefficients)
    assert guard_module.Affine(3).coefficients == (0, 0, 0, 0)
    with pytest.raises(TypeError):
        nd(3)
    with pytest.raises(FrozenInstanceError):
        a.offset = 0


def test_affine_keeps_legacy_keyword_and_subclass_result_contract():
    nd = importlib.import_module(PACKAGE + '.block_batching_guard_v2').AffineND

    class Legacy(guard_module.Affine):
        pass

    class Multiaxis(nd):
        pass

    assert type(Legacy(2).scale(scale=3)) is guard_module.Affine
    assert type(Multiaxis(2, (1, )).scale(value=3)) is nd
    # Refactoring must not add dimension validation or change zip truncation.
    assert Legacy(1, (1, 2)).add(nd(2, (3, ))).coefficients == (4, )


@pytest.mark.parametrize('grid,cv,preserve,capture,floor,expected,queries', [
    ((0, 1, 1), False, False, True, 99, 'EMPTY_GRID', []),
    ((2, 1, 1), True, False, True, 99, 'CV_GRID_NOT_DIVISIBLE', ['capture']),
    ((1, 1, 1), False, False, True, 99, 'GRID_NOT_REDUCED', ['capture']),
    ((6, 1, 1), False, False, True, 3, 'GRID_BELOW_CORE_FLOOR', ['capture', 'floor']),
    ((6, 1, 1), False, False, True, 2, 'CAPTURE_BASELINE', ['capture', 'floor']),
    ((6, 1, 1), False, False, None, 2, 'CAPTURE_QUERY_UNAVAILABLE', ['capture', 'floor']),
    ((6, 1, 1), False, False, False, 2, 'PROVEN_AT_RUNTIME', ['capture', 'floor', 'guard']),
    ((1, 1, 1), False, True, False, 1, 'PROVEN_AT_RUNTIME', ['capture', 'floor', 'guard']),
])
def test_dispatch_keeps_gate_order_and_schedule_grid(bundle_environment, monkeypatch, grid, cv, preserve, capture,
                                                     floor, expected, queries):
    env = bundle_environment
    env.wrapper._init_handles()
    env.wrapper.request = batching.BatchRequest(3, 0)
    env.report["cv_mode"] = "independent_batch_v1" if cv else "disabled"
    env.wrapper._preserve_grid = preserve
    calls = []

    def query_capture():
        calls.append('capture')
        return capture

    def query_floor():
        calls.append('floor')
        return floor

    def guard(args, original_grid):
        calls.append('guard')
        assert original_grid == grid
        return True, 'PROVEN_AT_RUNTIME'

    monkeypatch.setattr(batching, '_capture_state', query_capture)
    monkeypatch.setattr(batching, '_core_floor', query_floor)
    env.wrapper._guard = guard
    env.wrapper._dispatch(*grid, *env.launch[3:])
    report = env.wrapper.last_launch
    assert calls == queries
    assert report.get('reason', report['runtime_status']) == expected
    if expected == 'PROVEN_AT_RUNTIME':
        assert env.launches[-1][0] == 'candidate'
        assert report['batch_grid'] == (grid if preserve else ((grid[0] + 2) // 3, *grid[1:]))
        assert report['grouping'] == ('loop_iterations' if preserve else 'programs')
    elif expected == 'EMPTY_GRID':
        assert not env.launches
    else:
        assert env.launches[-1][0] == 'baseline'


def test_no_candidate_does_not_query_capture_floor_or_guard(bundle_environment, monkeypatch):
    env = bundle_environment
    env.wrapper.candidate = None
    env.report['reason'] = 'STATIC_REFUSAL'

    def forbidden(*args):
        raise AssertionError('unavailable candidate must not query launch prerequisites')

    monkeypatch.setattr(batching, '_capture_state', forbidden)
    monkeypatch.setattr(batching, '_core_floor', forbidden)
    env.wrapper._guard = forbidden
    env.wrapper._dispatch(*env.launch)
    assert env.wrapper.last_launch['reason'] == 'STATIC_REFUSAL'
    assert env.launches[-1][0] == 'baseline'


def test_core_query_failure_still_precedes_capture_refusal(bundle_environment, monkeypatch):
    env = bundle_environment
    monkeypatch.setattr(batching, '_capture_state', lambda: True)

    def invalid_floor():
        raise ValueError('invalid core configuration')

    monkeypatch.setattr(batching, '_core_floor', invalid_floor)
    with pytest.raises(ValueError, match='invalid core configuration'):
        env.wrapper._dispatch(*env.launch)
    assert not env.launches


def test_report_isolation_is_not_sneaked_into_structural_refactoring(bundle_environment):
    env = bundle_environment
    env.wrapper._dispatch(*env.launch)
    assert env.wrapper.last_launch['runtime_status'] == 'APPLIED'
    env.report['selected_axis'] = 1
    env.wrapper._dispatch(*env.launch)
    assert env.wrapper.last_launch['reason'] == 'GRID_NOT_REDUCED'


def test_candidate_execution_error_is_not_replayed_on_baseline(bundle_environment):
    env = bundle_environment

    def fail(*launch):
        raise RuntimeError('candidate launch failed')

    env.candidate.run = fail
    with pytest.raises(RuntimeError, match='candidate launch failed'):
        env.wrapper._dispatch(*env.launch)
    assert not env.launches
    assert env.wrapper.last_launch['runtime_status'] == 'NOT_LAUNCHED'


def test_owned_effect_codec_without_mlir_under_ubsan(tmp_path):
    import os
    from pathlib import Path
    import re
    import shlex
    import shutil
    import subprocess
    root = Path(__file__).resolve().parents[2]
    llvm = shutil.which('llvm-config')
    if os.getenv('LLVM_SYSPATH'):
        llvm = str(Path(os.environ['LLVM_SYSPATH']) / 'bin/llvm-config')
    if not llvm:
        for cache in (root.parents[1] / 'build').glob('*/CMakeCache.txt'):
            match = re.search(r'^LLVM_DIR:[^=]+=(.+)/lib/cmake/llvm$', cache.read_text(), re.M)
            if match:
                llvm = str(Path(match[1]) / 'bin/llvm-config')
                break
    compiler = os.getenv('CXX', 'g++')
    if not llvm or not Path(llvm).is_file() or not shutil.which(compiler):
        pytest.skip('owned-fact UBSan test needs a C++ compiler and LLVM development installation')
    include = subprocess.check_output([llvm, '--includedir'], text=True).strip()
    libs = shlex.split(subprocess.check_output([llvm, '--ldflags', '--libs', 'support', '--system-libs'], text=True))
    sources = [root / 'unittest/BlockBatching/EffectProjectionTest.cpp']
    sources += [root / 'lib/TritonToGraph/BlockBatching' / name for name in ['BatchExpr.cpp', 'BatchFactsJSON.cpp']]
    exe = tmp_path / 'effect_projection'
    command = [
        compiler, '-std=c++17', '-O1', '-g', '-fsanitize=undefined', '-fno-sanitize-recover=all',
        '-I' + str(root / 'include'), '-I' + include, *map(str, sources), *libs, '-o',
        str(exe)
    ]
    # Match the linked LLVM build; UBSan vptr instrumentation requires RTTI
    # symbols which a no-RTTI LLVMSupport intentionally does not export.
    if subprocess.check_output([llvm, '--has-rtti'], text=True).strip() == 'NO':
        command.insert(1, '-fno-rtti')
    subprocess.run(command, capture_output=True, text=True, check=True)
    result = subprocess.run([str(exe)], capture_output=True, text=True, check=True)
    assert 'owned effect projection regressions passed' in result.stdout
    assert not result.stderr
