# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Original-kernel correctness audit with explicit actual-variant reporting.

Use --require-applied to reject fallback-only device validation. Event timings
include host submission gaps; bench_block_batching_performance.py uses profiler
kernel durations for device performance comparisons.
"""
import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import torch_npu
import triton
from block_batching_kernels import (_rmsnorm_fwd_fused,
    _fused_situgluv2_fwd_triton_kernel, _fused_situgluv2_bwd_triton_kernel)


TOLERANCES = {torch.float32: (2e-4, 2e-5), torch.float16: (2e-3, 2e-3), torch.bfloat16: (2e-2, 2e-2)}


def measure(launch, repeats=3, samples=20):
    for _ in range(5):
        launch()
    torch.npu.synchronize()
    device, call = [], []
    for _ in range(repeats):
        events = [(torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
                  for _ in range(samples)]
        elapsed = []
        for start, end in events:
            start.record()
            before = time.perf_counter_ns()
            launch()
            elapsed.append((time.perf_counter_ns() - before) / 1000)
            end.record()
        torch.npu.synchronize()
        device.append([start.elapsed_time(end) * 1000 for start, end in events])
        call.append(elapsed)
    return {"method": "NPU event interval around one eager launch; includes submission gaps",
            "device_event_us": device, "host_enqueue_us": call,
            "device_event_round_medians_us": [statistics.median(x) for x in device],
            "host_enqueue_round_medians_us": [statistics.median(x) for x in call]}


def prepare_case(kind, rows, cols, dtype, factor=1, axis=0, eps=1e-6):
    torch.manual_seed(42)
    a = torch.randn((rows, cols), device="npu", dtype=dtype)
    b = torch.randn_like(a)
    options = dict(debug=True)
    if factor != 1:
        options.update(block_batching_factor=factor, block_batching_axis=axis)
    if kind == "rmsnorm":
        weight = torch.randn(cols, device="npu", dtype=torch.float32)
        out = torch.full_like(a, float("nan"))
        rrms = torch.full((rows,), float("nan"), device="npu", dtype=torch.float32)
        grid = (rows,)
        def launch():
            return _rmsnorm_fwd_fused[grid](a, out, weight, rrms, cols, cols, eps,
                                           D_ROUNDED=triton.next_power_of_2(cols), **options)
        inv = torch.rsqrt((a.float() ** 2).mean(1) + eps)
        expected = [(a.float() * inv[:, None] * weight).to(dtype), inv]
        outputs = [out, rrms]
    else:
        grid = (rows, triton.cdiv(cols, 128))
        out = torch.full_like(a, float("nan"))
        beta = linear_beta = 0.5
        if kind == "situ_fwd":
            def launch():
                return _fused_situgluv2_fwd_triton_kernel[grid](
                    a, b, out, cols, cols, 1, cols, 1, cols, 1,
                    beta, 1 / beta, linear_beta, 1 / linear_beta, BLOCK=128, **options)
            expected = [(beta * torch.tanh(a.float() / beta) * torch.sigmoid(a.float()) *
                         linear_beta * torch.tanh(b.float() / linear_beta)).to(dtype)]
            outputs = [out]
        else:
            dout = torch.randn_like(a)
            db = torch.full_like(b, float("nan"))
            def launch():
                return _fused_situgluv2_bwd_triton_kernel[grid](
                    a, b, dout, out, db, cols, cols, 1, cols, 1, cols, 1, cols, 1, cols, 1,
                    beta, 1 / beta, linear_beta, 1 / linear_beta, BLOCK=128, **options)
            # Independent autograd reference rather than copying the kernel derivative.
            ar, br = a.float().detach().requires_grad_(), b.float().detach().requires_grad_()
            reference = beta * torch.tanh(ar / beta) * torch.sigmoid(ar) * linear_beta * torch.tanh(br / linear_beta)
            expected = [x.to(dtype) for x in torch.autograd.grad(reference, (ar, br), dout.float())]
            outputs = [out, db]
    return {"launch": launch, "options": options, "outputs": outputs, "expected": expected, "grid": grid}


def run_case(kind, rows, cols, dtype, factor=1, axis=0, eps=1e-6, timing=False, require_applied=False):
    case = prepare_case(kind, rows, cols, dtype, factor, axis, eps)
    launch, outputs, expected, grid = (case[k] for k in ("launch", "outputs", "expected", "grid"))
    first_call_start = time.perf_counter()
    compiled = launch()
    torch.npu.synchronize()
    first_call_wall_ms = (time.perf_counter() - first_call_start) * 1000
    rtol, atol = TOLERANCES[dtype]
    for output, reference in zip(outputs, expected):
        output_rtol, output_atol = TOLERANCES[output.dtype]
        torch.testing.assert_close(output, reference, rtol=output_rtol, atol=output_atol)
    report = getattr(compiled.metadata, "block_batching", None)
    actual = getattr(compiled, "last_launch", {"runtime_status": "BASELINE_SELECTED"})
    if factor > 1:
        assert report is not None, "batching request did not return a report"
        if require_applied:
            assert actual["runtime_status"] == "APPLIED", (report, actual)
            assert actual["applied_factor"] == factor, actual
    result = {"kernel": kind, "shape": [rows, cols], "dtype": str(dtype), "eps": eps,
              "original_grid": list(grid), "factor": factor, "axis": axis,
              "strides": [cols, 1], "first_call_wall_ms": first_call_wall_ms,
              "constexpr": ({"D_ROUNDED": triton.next_power_of_2(cols)} if kind == "rmsnorm" else {"BLOCK": 128}),
              "scalars": ({"d": cols, "eps": eps} if kind == "rmsnorm" else
                          {"N": cols, "beta": 0.5, "inv_beta": 2.0, "linear_beta": 0.5, "inv_linear_beta": 2.0}),
              "correctness": True, "rtol": rtol, "atol": atol,
              "output_tolerances": [{"dtype": str(x.dtype), "rtol": TOLERANCES[x.dtype][0],
                                     "atol": TOLERANCES[x.dtype][1]} for x in outputs],
              "compile_report": report, "execution": actual,
              "max_abs_error": [float((x.float() - y.float()).abs().max()) for x, y in zip(outputs, expected)],
              "metadata": compiled.metadata._asdict()}
    if timing:
        result["timing"] = measure(launch)
    return result, compiled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--require-applied", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    environment = {"python": platform.python_version(), "torch": torch.__version__,
                   "torch_npu": torch_npu.__version__, "triton": triton.__version__,
                   "triton_path": triton.__file__, "device": torch.npu.get_device_name(0),
                   "fixture_sha256": hashlib.sha256(Path(__file__).with_name("block_batching_kernels.py").read_bytes()).hexdigest()}
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2))
    results = []
    for dtype in TOLERANCES:
        for kind in ("rmsnorm", "situ_fwd", "situ_bwd"):
            for rows, cols in ((10, 129), (128, 512)):
                for eps in ((1e-6, 10.0) if kind == "rmsnorm" else (1e-6,)):
                    case = f"{kind}_{rows}_{cols}_{str(dtype).split('.')[-1]}_{eps}"
                    try:
                        result, compiled = run_case(kind, rows, cols, dtype, args.factor,
                                                    0, eps, args.timing, args.require_applied)
                        for stage, content in compiled.asm.items():
                            if isinstance(content, str):
                                (args.output / f"{case}.{stage}").write_text(content)
                        result["case"] = case
                    except Exception as error:
                        result = {"case": case, "correctness": False, "error": repr(error)}
                    results.append(result)
                    (args.output / "results.json").write_text(json.dumps(results, indent=2, default=str))
                    print(json.dumps({key: value for key, value in result.items() if key not in ("metadata", "timing")}), flush=True)
    if any(not result["correctness"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
