# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import pytest
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["NR", "NC", "SX", "SY"])
def _symbolic_rows(X, Y, NR, NC, SX, SY, OTHER: tl.constexpr, M: tl.constexpr, D: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    row = p * M + tl.arange(0, M)
    col = tl.arange(0, D)
    live = row < NR
    values = tl.load(X + row[:, None] * SX + col[None, :], live[:, None] & (col[None, :] < NC), other=7.0)
    result = values.to(tl.float32) + p.to(tl.float32) + tl.num_programs(0).to(tl.float32)
    if OTHER:
        mask = tl.broadcast_to(live[:, None], (M, D))
    else:
        mask = live[:, None] & (col[None, :] < NC)
    tl.store(Y + row[:, None] * SY + col[None, :], result, mask)


@pytest.mark.parametrize("factor", [2, 3])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("other", [False, True])
def test_dynamic_strides_masks_other_and_cache(factor, dtype, other):
    seen = set()
    for nr, nc, sx, sy, grid in [(9, 5, 13, 19, 5), (9, 5, 16, 24, 4), (9, 5, 16, 24, 7), (1, 1, 8, 8, 1),
                                 (0, 5, 16, 24, 1), (0, 5, 16, 24, 0), (9, 5, 13, 19, 5)]:
        size = max(nr, grid * 2) + 8
        cpu_x = torch.arange(size * sx, dtype=torch.float32).reshape(size, sx).to(dtype)
        x = cpu_x.to('npu')
        y = torch.full((size, sy), -123.0, dtype=dtype, device='npu')
        kernel = _symbolic_rows[(grid, )](x, y, nr, nc, sx, sy, other, 2, 8, block_batching_factor=factor,
                                          block_batching_axis=0)
        torch.npu.synchronize()
        expected = torch.full((size, sy), -123.0, dtype=dtype)
        for row in range(min(nr, grid * 2)):
            for col in range(8 if other else nc):
                value = float(cpu_x[row, col]) if col < nc else 7.0
                expected[row, col] = value + row // 2 + grid
        torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(x.cpu(), cpu_x, rtol=0, atol=0)
        if grid:
            assert kernel.last_launch['runtime_status'] == 'APPLIED', kernel.block_batching
            plans = kernel.block_batching['layout']['accesses']
            assert all(p['rewritten'] and p['symbolic_stride'] for p in plans)
            assert len(kernel.block_batching['guard']['layout_checks']) == 2
            adapter = kernel.candidate.asm['ttadapter']
            assert 'triton_indirect' not in adapter
            assert 'linalg.transpose' not in adapter
            seen.add(kernel.candidate.hash)
        else:
            assert kernel.last_launch['runtime_status'] == 'EMPTY_GRID'
    # do_not_specialize keeps this dynamic-stride candidate reusable.
    assert len(seen) == 1


def test_dynamic_stride_guard_falls_back_then_recovers():
    x = torch.arange(128, device='npu', dtype=torch.float32).reshape(16, 8)
    y = torch.full((16, 8), -123.0, device='npu')
    for sx in (8, 0, 8):
        y.fill_(-123)
        kernel = _symbolic_rows[(3, )](x, y, 5, 5, sx, 8, False, 2, 8, block_batching_factor=3, block_batching_axis=0)
        torch.npu.synchronize()
        expected = torch.full((16, 8), -123.0)
        for row in range(5):
            expected[row, :5] = torch.arange(5) + row * sx + row // 2 + 3
        torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
        if sx:
            assert kernel.last_launch['runtime_status'] == 'APPLIED'
        else:
            assert kernel.last_launch['runtime_status'] == 'BASELINE_SELECTED'
            assert kernel.last_launch['reason'] == 'NON_POSITIVE_LAYOUT_STRIDE'
            assert kernel.last_launch['applied_factor'] == 1


def test_dynamic_stride_auto_and_semantic_have_identical_baseline():
    x = torch.arange(256, device='npu', dtype=torch.float32).reshape(16, 16)
    y = torch.full((16, 24), -123.0, device='npu')
    kernels, outputs = [], []
    for policy in ('auto', 'semantic_only', 'auto'):
        y.fill_(-123)
        kernel = _symbolic_rows[(5, )](x, y, 9, 5, 16, 24, False, 2, 8, block_batching_factor=3, block_batching_axis=0,
                                       block_batching_layout=policy)
        torch.npu.synchronize()
        assert kernel.last_launch['runtime_status'] == 'APPLIED'
        kernels.append(kernel)
        outputs.append(y.clone())
    assert kernels[0] is kernels[2]
    assert kernels[0].baseline.asm['ttir'] == kernels[1].baseline.asm['ttir']
    assert kernels[0].candidate.hash != kernels[1].candidate.hash
    assert 'layout_checks' not in kernels[1].block_batching['guard']
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    torch.testing.assert_close(outputs[0], outputs[2], rtol=0, atol=0)
