# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import json
import pytest
from triton._C.libtriton import ascend
from triton.backends.ascend.block_batching_guard import RuntimeGuard
from test_block_batching_rewrite import parse
from block_batching_symbolic_cases import symbolic_memory_ir

pytestmark = pytest.mark.backend("native")


@pytest.fixture(scope="session", autouse=True)
def assign_npu():
    yield


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("factor", [2, 3, 6])
@pytest.mark.parametrize("origin", [0, 2])
def test_symbolic_row_view_with_independent_strides(tmp_path, axis, factor, origin):
    module = parse(tmp_path, symbolic_memory_ir(axis, origin=origin))
    report = ascend.passes.ttir.rewrite_block_batching(module, factor, axis)
    assert report["compile_status"] == "REWRITTEN", report
    assert module.verify()
    plans = json.loads(report["layout_json"])["accesses"]
    assert all(p["action"] == "MemoryView" and p["rewritten"] and p["symbolic_stride"] for p in plans)
    assert all(p["reason"] == "PROVEN_SYMBOLIC_STRIDE_CHAIN" for p in plans)
    assert plans[0]["element_stride"] != plans[1]["element_stride"]
    guard = json.loads(report["guard_json"])
    guard["axis"] = report["selected_axis"]
    assert len(guard["layout_checks"]) == 2
    grid = [1, 1, 1]
    grid[axis] = 5
    runtime = RuntimeGuard(guard, factor)
    assert runtime([1 << 32, 1 << 45, 9, 4, 16, 20], grid)[0]
    assert runtime([1 << 32, 1 << 45, 9, 3, 8, 12], grid)[0]
    assert not runtime([1 << 32, 1 << 45, 9, 4, 16, 2], grid)[0]
    assert not runtime([1 << 32, 1 << 32, 9, 4, 16, 16], grid)[0]
    text = str(module)
    assert f"tensor<{factor*2}x4x!tt.ptr<f32>>" in text
    assert f"tensor<{factor}x2x4xf32>" in text
    assert "tt.trans" not in text
    assert "7.000000e+00" in text


def test_affine_uniform_parameter_is_not_a_kernel_specific_pattern(tmp_path):
    module = parse(tmp_path, symbolic_memory_ir(mode="affine_parameter"))
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report["compile_status"] == "REWRITTEN", report
    assert all(p["rewritten"] for p in json.loads(report["layout_json"])["accesses"])
    guard = json.loads(report["guard_json"])
    guard["axis"] = report["selected_axis"]
    assert RuntimeGuard(guard, 3)([1 << 32, 1 << 45, 9, 4, 4, 12], (5, 1, 1))[0]


@pytest.mark.parametrize("mode", ["mismatch", "parameter_product"])
def test_unproved_x_stride_keeps_only_that_access(tmp_path, mode):
    module = parse(tmp_path, symbolic_memory_ir(mode=mode))
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report["compile_status"] == "REWRITTEN", report
    plans = json.loads(report["layout_json"])["accesses"]
    assert not plans[0]["rewritten"] and plans[0]["reason"] == "STRIDE_CHAIN_UNPROVEN"
    assert plans[1]["rewritten"]
    assert [g["access_id"] for g in json.loads(report["guard_json"])["layout_checks"]] == [1]
    assert module.verify()


def test_semantic_only_does_not_publish_layout_guard(tmp_path):
    module = parse(tmp_path, symbolic_memory_ir())
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0, "semantic_only")
    assert report["compile_status"] == "REWRITTEN", report
    assert "layout_checks" not in json.loads(report["guard_json"])
    assert not any(p["rewritten"] for p in json.loads(report["layout_json"])["accesses"])


def test_symbolic_column_layout_is_not_silently_enabled(tmp_path):
    source = symbolic_memory_ir().replace("%xa = tt.addptr",
                                          "%xa0 = tt.addptr").replace("%ya = tt.addptr", "%ya0 = tt.addptr")
    source = source.replace(
        "    %other =",
        """    %xa = tt.trans %xa0 {order = array<i32: 1, 0>} : tensor<2x4x!tt.ptr<f32>> -> tensor<4x2x!tt.ptr<f32>>
    %ya = tt.trans %ya0 {order = array<i32: 1, 0>} : tensor<2x4x!tt.ptr<f32>> -> tensor<4x2x!tt.ptr<f32>>
    %mask2 = tt.trans %mask {order = array<i32: 1, 0>} : tensor<2x4xi1> -> tensor<4x2xi1>
    %other =""")
    start = source.index("    %other =")
    source = source[:start] + source[start:].replace("2x4x", "4x2x").replace("%mask,", "%mask2,").replace(
        "%mask :", "%mask2 :")
    module = parse(tmp_path, source)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report["compile_status"] == "REWRITTEN", report
    assert all(not p["rewritten"] and p["reason"] == "SYMBOLIC_LAYOUT_BRIDGE_UNSUPPORTED"
               for p in json.loads(report["layout_json"])["accesses"])
    assert "layout_checks" not in json.loads(report["guard_json"])
    assert module.verify()
