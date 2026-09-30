"""Opt-in row-tile reuse of the original Q2 RMSNorm DSL, with N held exact."""
import hashlib
import json
import os
from pathlib import Path

import pytest
import triton
from _rmsnorm_kernel import _rmsnorm_infer_kernel


@pytest.mark.parametrize('dtype_name', ['float32', 'float16', 'bfloat16'])
@pytest.mark.parametrize('padded', [False, True])
@pytest.mark.parametrize('rows,columns,programs,old,new', [
    (17, 384, 3, 4, 8),
    (17, 385, 3, 8, 4),
    (1, 31, 8, 2, 4),
    (33, 257, 1, 4, 2),
])
def test_rmsnorm_m_reuses_fixed_n(dtype_name, padded, rows, columns, programs, old, new):
    if os.environ.get('RUN_NPU_REUSE_TESTS') != '1': pytest.skip('set RUN_NPU_REUSE_TESTS=1')
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(0)
    dtype = getattr(torch, dtype_name)
    generator = torch.Generator().manual_seed(1843)
    x_cpu = (torch.randn((rows, columns), generator=generator) * 1.7).to(dtype)
    w_cpu = torch.randn((columns, ), generator=generator).to(dtype)
    x_pitch = columns + (16 if padded else 0)
    y_pitch = columns + (32 if padded else 0)
    x_back = torch.full((rows, x_pitch), -777., dtype=dtype, device='npu')
    y_back = torch.full((rows, y_pitch), -777., dtype=dtype, device='npu')
    x = x_back[:, :columns]
    y = y_back[:, :columns]
    x.copy_(x_cpu)
    w = w_cpu.to('npu')
    reference = (x_cpu.float() * torch.rsqrt(x_cpu.float().square().mean(1, keepdim=True) + 1e-5) *
                 w_cpu.float()).to(dtype)
    kernel = triton.jit(_rmsnorm_infer_kernel.fn)
    calls = []

    def grid(meta):
        calls.append((meta['BLOCK_SIZE_M'], meta['BLOCK_SIZE_N']))
        return (programs, )

    launched = []
    outputs = []
    for m, n in ((old, 128), (new, 128), (new, 64)):
        y_back.fill_(-777.)
        result = kernel[grid](x, y, w, x_pitch, y_pitch, rows, columns, 1e-5, m, n)
        torch.npu.synchronize()
        actual = y_back.cpu()
        torch.testing.assert_close(actual[:, :columns], reference, rtol=4e-5 if dtype_name == 'float32' else .02,
                                   atol=4e-6 if dtype_name == 'float32' else .02)
        assert torch.all(actual[:, columns:] == -777)
        assert torch.equal(x_back[:, :columns].cpu(), x_cpu)
        assert torch.all(x_back[:, columns:] == -777)
        outputs.append(hashlib.sha256(actual.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
        launched.append({'hash': result.hash, 'M': result.src.constants[(8, )], 'N': result.src.constants[(9, )]})
    assert calls == [(old, 128), (new, 128), (new, 64)]
    if os.environ.get('TRITON_ASCEND_ENABLE_DYNAMIC_REUSE') == '1':
        assert launched[0]['hash'] == launched[1]['hash']
        assert launched[1]['M'] == old
    else:
        assert launched[1]['M'] == new
    assert launched[2]['N'] == 64 and launched[2]['hash'] != launched[0]['hash']
    if os.environ.get('RMS_REUSE_REPORT'):
        Path(os.environ['RMS_REUSE_REPORT']).write_text(
            json.dumps({'outputs': outputs, 'launches': launched, 'calls': calls}, indent=2))
