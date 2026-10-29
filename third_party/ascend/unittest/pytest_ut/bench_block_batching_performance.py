# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Compare fixed factors using MSPTI device timestamps, retaining per-task samples.

Requires mspti and libmspti.so in LD_PRELOAD before Python starts. Collection
follows torch_npu._inductor.profiler.mspti_batch_benchmark; the caller performs
five warmup launches before collecting active samples here.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch
import torch_npu
import triton
from triton import knobs
from bench_block_batching import prepare_case, TOLERANCES


def profile_kernel(launch, destination, samples):
    """Collect active device tasks with MSPTI after the caller's warmup."""
    from mspti import KernelMonitor

    if "libmspti.so" not in os.getenv("LD_PRELOAD", ""):
        raise RuntimeError("Set libmspti.so in LD_PRELOAD before starting Python to use MSPTI.")
    rows = []

    def callback(data):
        rows.append({
            "Name": data.name, "Start(ns)": data.start, "End(ns)": data.end, "Duration(us)":
            (data.end - data.start) / 1000
        })

    monitor = KernelMonitor()
    torch.npu.synchronize()
    monitor.start(callback)
    try:
        for _ in range(samples):
            launch()
    finally:
        try:
            torch.npu.synchronize()
        finally:
            monitor.stop()
    destination.mkdir(parents=True, exist_ok=True)
    raw_csv = destination / "mspti_kernel_details.csv"
    with raw_csv.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["Name", "Start(ns)", "End(ns)", "Duration(us)"])
        writer.writeheader()
        writer.writerows(rows)
    if len(rows) != samples:
        raise RuntimeError(f"expected exactly one device task per launch ({samples}), got {len(rows)}: {raw_csv}")
    durations = [float(row["Duration(us)"]) for row in rows]
    if not all(math.isfinite(t) and t > 0 for t in durations):
        raise RuntimeError("invalid device duration")
    return {
        "device_kernel_us": durations, "median_us": statistics.median(durations), "mean_us": statistics.mean(durations),
        "measurement_backend": "mspti", "kernel_names": sorted({row["Name"]
                                                                for row in rows}), "raw_csv": str(raw_csv)
    }


def measure_host(launch, samples):
    enqueue = []
    torch.npu.synchronize()
    begin = time.perf_counter_ns()
    for _ in range(samples):
        start = time.perf_counter_ns()
        launch()
        enqueue.append((time.perf_counter_ns() - start) / 1000)
    torch.npu.synchronize()
    return {
        "enqueue_us": enqueue, "median_enqueue_us": statistics.median(enqueue), "synchronized_batch_per_call_us":
        (time.perf_counter_ns() - begin) / 1000 / samples
    }


def run(kind, rows, cols, dtype, factor, axis, destination, rounds, samples):
    case = prepare_case(kind, rows, cols, dtype, 1, axis)
    launch, options = case["launch"], case["options"]
    options.update(block_batching_factor=factor, block_batching_axis=axis)
    candidate = launch()
    torch.npu.synchronize()
    actual = candidate.last_launch
    if actual["runtime_status"] != "APPLIED" or actual["applied_factor"] != factor:
        raise RuntimeError(f"candidate was not applied: {candidate.block_batching}, {actual}")
    for output, expected in zip(case["outputs"], case["expected"]):
        rtol, atol = TOLERANCES[output.dtype]
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)
    results = {
        "kind": kind, "shape": [rows, cols], "dtype": str(dtype), "factor": factor, "axis": axis, "launch":
        dict(actual), "candidate_hash": candidate.hash, "bundle_baseline_hash": candidate.baseline.hash,
        "backend_report": candidate.block_batching, "rounds": [], "seed": 42, "torch": torch.__version__, "torch_npu":
        torch_npu.__version__, "triton": triton.__version__, "triton_path": triton.__file__, "device":
        torch.npu.get_device_name(0), "taskqueue": knobs.ascend.enable_taskqueue, "fixture_sha256":
        hashlib.sha256(Path(__file__).with_name("block_batching_kernels.py").read_bytes()).hexdigest(),
        "samples_per_variant_per_round": samples, "warmup_per_variant_per_round": 5, "measurement_backend": "mspti",
        "bundle_baseline_ttir_sha256": hashlib.sha256(candidate.baseline.asm["ttir"].encode()).hexdigest()
    }
    for index in range(rounds):
        order = [1, factor] if index % 2 == 0 else [factor, 1]
        record = {}
        for selected_factor in order:
            options.update(block_batching_factor=selected_factor,
                           block_batching_axis=axis if selected_factor > 1 else -1)
            compiled = None
            for _ in range(5):
                compiled = launch()
            torch.npu.synchronize()
            if selected_factor > 1:
                assert compiled.last_launch["runtime_status"] == "APPLIED", compiled.last_launch
            else:
                results["standalone_baseline_hash"] = compiled.hash
                results["standalone_baseline_ttir_sha256"] = hashlib.sha256(compiled.asm["ttir"].encode()).hexdigest()
                results["baseline_ttir_identical"] = (
                    results["standalone_baseline_ttir_sha256"] == results["bundle_baseline_ttir_sha256"])
            prefix = destination / f"round{index}_factor{selected_factor}"
            measured = profile_kernel(launch, prefix, samples)
            measured["host"] = measure_host(launch, samples)
            for output, expected in zip(case["outputs"], case["expected"]):
                rtol, atol = TOLERANCES[output.dtype]
                torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)
            if selected_factor > 1:
                assert compiled.last_launch["runtime_status"] == "APPLIED", compiled.last_launch
            record[str(selected_factor)] = measured
        record["speedup"] = record["1"]["median_us"] / record[str(factor)]["median_us"]
        results["rounds"].append(record)
        results["guard_cache_info"] = candidate._guard._prove.cache_info()._asdict()
        (destination / "results.json").write_text(json.dumps(results, indent=2, default=str))
    results["median_speedup"] = statistics.median(r["speedup"] for r in results["rounds"])
    (destination / "results.json").write_text(json.dumps(results, indent=2, default=str))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument("--axis", type=int, default=0)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=512)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--kinds", nargs="+", default=["rmsnorm", "situ_fwd", "situ_bwd"])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    if args.rounds < 3 or args.samples < 5 or args.factor < 2:
        parser.error("require rounds>=3, samples>=5, factor>=2")
    args.output.mkdir(parents=True, exist_ok=True)
    for kind in args.kinds:
        destination = args.output / kind
        destination.mkdir(parents=True, exist_ok=True)
        result = run(kind, args.rows, args.cols, getattr(torch, args.dtype), args.factor, args.axis, destination,
                     args.rounds, args.samples)
        print(
            json.dumps({
                "kind": kind, "median_speedup": result["median_speedup"], "round_speedups":
                [r["speedup"] for r in result["rounds"]]
            }), flush=True)


if __name__ == "__main__":
    main()
