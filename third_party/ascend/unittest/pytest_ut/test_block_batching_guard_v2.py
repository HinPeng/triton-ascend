# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import importlib
from pathlib import Path
import random
import sys
import types

import pytest

pytestmark = pytest.mark.backend("native")
PACKAGE = "_batch_byte_proof_test"
if PACKAGE not in sys.modules:
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(Path(__file__).parents[2] / "backend")]
    sys.modules[PACKAGE] = package
module = importlib.import_module(PACKAGE + ".block_batching_guard")


def const(value, bits=64):
    return ["const", value, bits]


def arg(index, bits=64):
    return ["arg", index, bits]


def pid(axis):
    return ["pid", axis, 32]


def coord(axis, size):
    return ["coord", axis, size, 32]


def add(a, b, bits=64):
    return ["addi", bits, a, b]


def mul(a, b, bits=64):
    return ["muli", bits, a, b]


def matrix_guard(rows=2, columns=4, width=4):
    row = add(mul(pid(0), const(rows)), coord(0, rows))
    col = add(mul(pid(1), const(columns)), coord(1, columns))
    offset = mul(add(mul(row, arg(4)), col), const(width))
    mask = ["and", ["slt", row, arg(2, 32), 32], ["slt", col, arg(3, 32), 32]]
    return {
        "version":
        2, "axis":
        0, "accesses": [{
            "root": root, "shape": [rows, columns], "offset": offset, "mask": mask, "bytes": width, "write": write,
            "pointer_checks": [offset]
        } for root, write in [(0, False), (1, True)]]
    }


@pytest.mark.parametrize("width", [1, 2, 4, 8])
@pytest.mark.parametrize("rows,cols", [(1, 1), (3, 3), (17, 129)])
def test_two_independent_local_axes_and_tail_masks(width, rows, cols):
    spec = matrix_guard(2, 128, width)
    grid = ((rows + 1) // 2, (cols + 127) // 128, 1)
    assert module.check_guard(spec, [1 << 32, 1 << 45, rows, cols, cols], grid, 3)[0]
    if rows > 1 and cols > 1:
        assert not module.check_guard(spec, [1 << 32, 1 << 45, rows, cols, cols - 1], grid, 3)[0]


def test_byte_ranges_detect_mixed_width_aliases():
    spec = {
        "version":
        2, "axis":
        0, "accesses": [{"root": 0, "shape": [], "offset": const(0), "mask": None, "write": False, "bytes": 8},
                        {"root": 1, "shape": [], "offset": const(0), "mask": None, "write": True, "bytes": 4}]
    }
    assert module.check_guard(spec, [10000, 10004], (1, 1, 1), 3) == (False, "WRITE_ALIAS")
    assert module.check_guard(spec, [10000, 10008], (1, 1, 1), 3)[0]


def test_mask_cannot_hide_source_i32_wraparound():
    index = add(mul(pid(0), const(4, 32), 32), coord(0, 4), 32)
    wide = mul(add(mul(pid(0), const(4)), coord(0, 4)), const(4))
    spec = {
        "version":
        2, "axis":
        0, "accesses":
        [{"root": 0, "shape": [4], "offset": wide, "mask": ["slt", index, arg(1, 32), 32], "write": True, "bytes": 4}]
    }
    assert module.check_guard(spec, [1 << 40, 1], ((1 << 30) + 1, 1, 1), 3) == (False, "INDEX_OVERFLOW")


def test_every_pointer_integer_intermediate_is_checked():
    spec = {
        "version":
        2, "axis":
        0, "accesses": [{
            "root": 0, "shape": [], "offset": const(0), "mask": None, "write": False, "bytes": 4, "pointer_checks":
            [arg(1)]
        }]
    }
    guard = module.RuntimeGuard(spec, 3)
    assert guard([10000, 0], (1, 1, 1))[0]
    assert guard([10000, -20000], (1, 1, 1)) == (False, "INDEX_OVERFLOW")


def test_eq_zero_proves_a_single_scalar_writer():
    spec = {
        "version":
        2, "axis":
        0, "accesses":
        [{"root": 0, "shape": [], "offset": const(0), "mask": ["eq", pid(0), const(0), 32], "write": True, "bytes": 4}]
    }
    assert module.check_guard(spec, [10000], (13, 1, 1), 3)[0]
    spec["accesses"][0]["mask"] = None
    assert not module.check_guard(spec, [10000], (13, 1, 1), 3)[0]


def test_randomized_accepted_boxes_have_no_overlapping_original_writes():
    rng = random.Random(107)
    accepted = 0
    for _ in range(300):
        tile_r, tile_c = rng.randint(1, 3), rng.randint(1, 4)
        g0, g1 = rng.randint(1, 3), rng.randint(1, 3)
        rows, cols = rng.randint(1, g0 * tile_r), rng.randint(1, g1 * tile_c)
        stride, width = rng.randint(0, cols + 2), rng.choice([1, 2, 4, 8])
        spec = matrix_guard(tile_r, tile_c, width)
        ok, _ = module.check_guard(spec, [1 << 32, 1 << 45, rows, cols, stride], (g0, g1, 1), 3)
        if not ok:
            continue
        accepted += 1
        written = set()
        for gr in range(g0):
            for gc in range(g1):
                for local_r in range(tile_r):
                    for local_c in range(tile_c):
                        row, col = gr * tile_r + local_r, gc * tile_c + local_c
                        if row >= rows or col >= cols:
                            continue
                        address = (row * stride + col) * width
                        bytes_ = set(range(address, address + width))
                        assert written.isdisjoint(bytes_)
                        written.update(bytes_)
    assert accepted > 40


def loop_guard():
    return {
        "version":
        2, "axis":
        0, "accesses": [{
            "root":
            0, "shape": [], "offset":
            mul(add(mul(pid(0), const(8)), ["iv", 0, 32]), const(4)), "mask":
            None, "bytes":
            4, "write":
            False, "loops": [{"id": 0, "bits": 32, "lower": const(0, 32), "upper": arg(2, 32), "step": const(1, 32)}]
        }, {"root": 1, "shape": [], "offset": mul(pid(0), const(4)), "mask": None, "bytes": 4, "write": True}]
    }


@pytest.mark.parametrize("iterations", [0, 1, 4])
def test_readonly_loop_has_a_separate_induction_coordinate(iterations):
    assert module.check_guard(loop_guard(), [10000, 100000, iterations], (4, 1, 1), 3)[0]


def test_zero_trip_loop_does_not_hide_an_outside_writer():
    spec = loop_guard()
    spec["accesses"][1]["offset"] = const(0)
    assert module.check_guard(spec, [10000, 100000, 0], (4, 1, 1), 3) == (False, "OVERLAPPING_BLOCK_WRITES")


def test_loop_bounds_are_included_in_proof_cache_dependencies():
    guard = module.RuntimeGuard(loop_guard(), 3)
    assert guard([10000, 100000, 4], (2, 1, 1))[0]
    assert guard([10000, 100000, 25000], (2, 1, 1)) == (False, "WRITE_ALIAS")
    assert guard([10000, 100000, 0], (2, 1, 1))[0]


def test_loop_induction_increment_cannot_overflow():
    spec = loop_guard()
    spec["accesses"][0]["loops"][0].update(lower=const((1 << 31) - 2, 32), upper=const((1 << 31) - 1, 32),
                                           step=const(4, 32))
    assert module.check_guard(spec, [10000, 100000, 4], (2, 1, 1), 3) == (False, "INDEX_OVERFLOW")


@pytest.mark.parametrize("factor", [2, 3, 6])
def test_volatile_unique_reads_and_element_tails(factor):
    index = add(mul(pid(0), const(4)), coord(0, 4))
    spec = {
        "version":
        2, "axis":
        0, "accesses": [{
            "root": 0, "shape": [4], "offset": mul(index, const(4)), "mask": ["slt", index,
                                                                              arg(1, 32), 64], "write": False, "bytes":
            4, "volatile": True
        }]
    }
    assert module.check_guard(spec, [1 << 32, 13], (4, 1, 1), factor)[0]
    # A second grid direction must not silently share the volatile read.
    assert module.check_guard(spec, [1 << 32, 13], (4, 2, 1), factor) == (False, "OVERLAPPING_VOLATILE_READS")


def test_shared_volatile_load_requires_no_duplicate_original_invocations():
    access = {"root": 0, "shape": [], "offset": const(0), "mask": None, "write": False, "bytes": 4, "volatile": True}
    spec = {"version": 2, "axis": 0, "accesses": [access]}
    assert module.check_guard(spec, [10000], (1, 1, 1), 3)[0]
    assert module.check_guard(spec, [10000], (2, 1, 1), 3) == (False, "OVERLAPPING_VOLATILE_READS")
    access["volatile"] = False
    assert module.check_guard(spec, [10000], (2, 1, 1), 3)[0]


def test_volatile_is_still_a_read_for_alias_checks():
    access = {
        "root": 0, "shape": [], "offset": mul(pid(0), const(4)), "mask": None, "write": False, "bytes": 4, "volatile":
        True
    }
    spec = {"version": 2, "axis": 0, "accesses": [access, dict(access)]}
    assert module.check_guard(spec, [10000], (7, 1, 1), 3)[0]
    spec["accesses"][1] = dict(access, write=True, volatile=False)
    assert module.check_guard(spec, [10000], (7, 1, 1), 3) == (False, "WRITE_ALIAS")


def test_volatile_cannot_hide_arithmetic_overflow_or_invalid_schema():
    access = {
        "root": 0, "shape": [], "offset": mul(pid(0), const(4, 32), 32), "mask": ["false"], "write": False, "bytes": 4,
        "volatile": True
    }
    spec = {"version": 2, "axis": 0, "accesses": [access]}
    assert module.check_guard(spec, [1 << 40], (1 << 30, 1, 1), 3) == (False, "INDEX_OVERFLOW")
    access["volatile"] = 1
    assert module.check_guard(spec, [1 << 40], (1, 1, 1), 3) == (False, "INVALID_MEMORY_ARGUMENT")
