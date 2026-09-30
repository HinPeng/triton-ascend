"""Opt-in NPU proof for the unchanged vLLM penalties DSL."""
import os

import pytest
import triton
from _penalties_kernel import apply_all_penalties_kernel


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("old,new", [(128, 256), (256, 128)])
@pytest.mark.parametrize("programs", [1, 2, 8])
def test_penalties_preserved_outer_reuses_binary(strided, old, new, programs):
    if os.environ.get("RUN_NPU_REUSE_TESTS") != "1":
        pytest.skip("set RUN_NPU_REUSE_TESTS=1")
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(0)
    kernel = triton.jit(apply_all_penalties_kernel.fn)
    rows, columns, stride = 3, 513, 2 if strided else 1
    cpu = ((torch.arange(rows * columns).reshape(rows, columns) % 17) - 8).float() / 4
    prompt = torch.arange(columns)[None, :].expand(rows, -1) % 3 == 0
    counts = torch.arange(rows * columns).reshape(rows, columns) % 4
    output = counts > 0
    repeat = torch.tensor([2., .5, 1.])
    frequency = torch.tensor([.25, -.5, 0.])
    presence = torch.tensor([.5, -.25, 0.])
    factor = torch.where(prompt | output, repeat[:, None], 1.)
    expected = cpu * torch.where(cpu > 0, 1. / factor, factor)
    expected -= frequency[:, None] * counts
    expected -= presence[:, None] * output
    device_inputs = [v.to('npu') for v in (prompt, output, counts.to(torch.int32), repeat, frequency, presence)]
    backing = torch.full((rows, columns * stride + 16), -777., device='npu')
    logits = backing[:, :columns * stride:stride]
    calls = []

    def grid(meta):
        calls.append(meta['BLOCK_SIZE'])
        return (programs, )

    results = []
    for tile in (old, new):
        logits.copy_(cpu)
        desc = kernel[grid](logits, *device_inputs, rows, columns, logits.stride(0), logits.stride(1), columns, 1,
                            columns, 1, columns, 1, tile)
        torch.npu.synchronize()
        reference = torch.full(backing.shape, -777.)
        reference[:, :columns * stride:stride] = expected
        assert torch.equal(backing.cpu(), reference)
        results.append(desc)
    assert calls == [old, new]
    assert results[0].hash == results[1].hash
    assert results[1].src.constants[(17, )] == old
