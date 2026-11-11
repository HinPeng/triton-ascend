"""External floating input policies with exceptional values and partial groups."""
import pytest
import torch
import triton
import triton.language as tl
from triton.language.extra.cann.libdevice import pow as device_pow, tanh


@triton.jit
def external_inputs(X, Y, O, N, B: tl.constexpr, POW: tl.constexpr):
    offsets = tl.program_id(0) * B + tl.arange(0, B)
    valid = offsets < N
    x = tl.load(X + offsets, valid, other=0)
    if POW:
        y = tl.load(Y + offsets, valid, other=0)
        result = device_pow(x, y)
    else:
        result = tanh(x)
    tl.store(O + offsets, result, valid)


@pytest.mark.parametrize("power", [False, True])
@pytest.mark.parametrize("blocks", [128, 129])
def test_external_inputs_preserve_exceptional_values_and_guards(power, blocks):
    width, n = 128, blocks * 128 - 3
    values = torch.tensor([-float("inf"), -2., -0., 0., .5, 2., float("inf"), float("nan"), -2., 0.])
    exponents = torch.tensor([2., 3., 1., 2., .5, -1., 0., 2., .5, 0.])
    x_cpu = values.repeat(triton.cdiv(n, len(values)))[:n]
    y_cpu = exponents.repeat(triton.cdiv(n, len(exponents)))[:n]
    x, y = x_cpu.npu(), y_cpu.npu()
    storage = [torch.full((n + 32,), -123., device="npu") for _ in range(2)]
    outputs = [value[16:16 + n] for value in storage]
    kernels = [external_inputs[(blocks,)](x, y, out, n, width, power,
                block_batching_factor=factor, num_warps=4, num_stages=1)
               for factor, out in zip((1, 2), outputs)]
    torch.npu.synchronize()
    expected = torch.pow(x_cpu, y_cpu) if power else torch.tanh(x_cpu)
    for out, allocation in zip(outputs, storage):
        torch.testing.assert_close(out.cpu(), expected, rtol=2e-4, atol=2e-5, equal_nan=True)
        assert (allocation[:16].cpu() == -123.).all()
        assert (allocation[16 + n:].cpu() == -123.).all()
    torch.testing.assert_close(outputs[1].cpu(), outputs[0].cpu(), rtol=0, atol=0, equal_nan=True)
    zeros = torch.isfinite(expected) & (expected == 0)
    assert torch.equal(torch.signbit(outputs[1].cpu()[zeros]), torch.signbit(outputs[0].cpu()[zeros]))
    assert kernels[1].last_launch["applied_factor"] == 2
    assert kernels[0].kernel == kernels[1].baseline.kernel
