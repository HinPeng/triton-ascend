"""Paired native-loading gate, run on an otherwise idle validation device."""
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from test_npu import pointwise
from triton.backends.ascend.reuse.context import exact_request_scope

# Fix acceptance before measuring. This is a qualification criterion, not a
# claim about application latency or a production performance guarantee.
MAX_P99_RATIO = 1.20
MAX_STALL_MS = 1.0


def quantiles(values):
    values = sorted(values)
    return {
        "p50_ms": values[len(values) // 2], "p95_ms": values[int(len(values) * .95)], "p99_ms":
        values[int(len(values) * .99)], "max_ms": values[-1], "samples": len(values)
    }


def main():
    torch.npu.set_device(0)
    a = torch.arange(5000, device="npu", dtype=torch.float32) / 4
    b, out = torch.ones_like(a), torch.empty_like(a)
    with exact_request_scope():
        ready = pointwise[(3, )](a, b, out, .5, 5000, 5, 1024)
        cold = [
            pointwise.warmup(a, b, out, .5, 5000, work, tile, grid=(3, ))
            for tile in (128, 256, 512, 2048)
            for work in (1, (5000 + tile - 1) // tile)
        ]
    launch = ready[(3, 1, 1)]
    expected = a * .5 + b

    def sample():
        out.fill_(-9)
        start = time.perf_counter_ns()
        launch(a, b, out, .5, 5000, 5, 1024)
        torch.npu.synchronize()
        return (time.perf_counter_ns() - start) / 1e6

    for _ in range(50):
        sample()
    baseline = quantiles([sample() for _ in range(500)])
    barrier = threading.Barrier(2)

    def load():
        torch.npu.set_device(0)
        barrier.wait()
        for kernel in cold:
            assert not kernel._initialization_complete
            kernel._init_handles()
            assert kernel._initialization_complete and kernel.module and kernel.function

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(load)
        barrier.wait()
        concurrent = []
        while len(concurrent) < 500 or not future.done():
            concurrent.append(sample())
            if len(concurrent) >= 20000:
                break
        future.result(timeout=60)
    assert torch.equal(out, expected)
    concurrent = quantiles(concurrent)
    result = {
        "baseline": baseline, "concurrent": concurrent, "correct": True, "cold_objects": len(cold), "thresholds":
        {"p99_ratio": MAX_P99_RATIO, "max_stall_ms": MAX_STALL_MS}
    }
    result['qualified'] = (concurrent['p99_ms'] <= baseline['p99_ms'] * MAX_P99_RATIO
                           and concurrent['max_ms'] <= MAX_STALL_MS)
    Path(sys.argv[1]).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
