"""Descriptive synchronized host latency; use separate off/on processes."""
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
from qualify_loading import quantiles
from test_analysis import scalar_store
from test_npu import pointwise


@triton.jit
def conservative(a, out, N: tl.constexpr, B: tl.constexpr):
    offsets = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(a + offsets, offsets < N, other=0)
    tl.store(out + offsets, tl.abs(x), offsets < N)


def main():
    torch.npu.set_device(0)
    a = torch.ones(5000, device='npu')
    b, out = a.clone(), torch.empty_like(a)
    integer = torch.empty(16, device='npu', dtype=torch.int32)
    cases = {
        'exact': lambda: pointwise[(3, )](a, b, out, .5, 5000, 5, 1024),
        'compatible': lambda: pointwise[(3, )](a, b, out, .5, 5000, 3, 2048),
        'dynamic': lambda: scalar_store[(7, )](integer, 3, 7),
        'unknown': lambda: conservative[(5, )](a, out, 5000, 1024),
    }
    results = {}
    for name, call in cases.items():
        for _ in range(25):
            call()
        torch.npu.synchronize()
        elapsed = []
        for _ in range(500):
            start = time.perf_counter_ns()
            call()
            torch.npu.synchronize()
            elapsed.append((time.perf_counter_ns() - start) / 1e6)
        results[name] = quantiles(elapsed)
    from triton.backends.ascend.reuse.bridge import diagnostics
    report = {
        'mode': os.environ.get('TRITON_ASCEND_ENABLE_DYNAMIC_REUSE'), 'latency': results, 'runtime': diagnostics()
    }
    Path(sys.argv[1]).write_text(json.dumps(report, indent=2))
    print(json.dumps(results))


if __name__ == '__main__':
    main()
