# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Compare original kernels with a fixed manual factor, retaining raw profiler data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
import torch_npu
import triton
from triton import knobs
from bench_block_batching import prepare_case, TOLERANCES


def profile_kernel(launch, destination, samples):
    """Read device task duration, with no host-submission interval in the metric."""
    from torch_npu import profiler as p
    with p.profile(activities=[p.ProfilerActivity.NPU],
                   on_trace_ready=p.tensorboard_trace_handler(str(destination)),
                   record_shapes=False, profile_memory=False, with_stack=False) as prof:
        for _ in range(samples):
            launch()
            torch.npu.synchronize()
    files = list(destination.rglob("kernel_details.csv"))
    if len(files) != 1:
        raise RuntimeError(f"expected one kernel_details.csv in {destination}, got {files}")
    with files[0].open() as source:
        rows = list(csv.DictReader(source))
    if len(rows) != samples:
        raise RuntimeError(f"expected {samples} device tasks, got {len(rows)}: {files[0]}")
    durations = [float(row["Duration(us)"]) for row in rows]
    if not all(t > 0 for t in durations):
        raise RuntimeError("invalid device duration")
    return {"device_kernel_us": durations, "median_us": statistics.median(durations),
            "kernel_names": sorted({row["Name"] for row in rows}), "raw_csv": str(files[0])}


def measure_host(launch, samples):
    enqueue = []
    torch.npu.synchronize()
    begin = time.perf_counter_ns()
    for _ in range(samples):
        start = time.perf_counter_ns()
        launch()
        enqueue.append((time.perf_counter_ns() - start) / 1000)
    torch.npu.synchronize()
    return {"enqueue_us": enqueue, "median_enqueue_us": statistics.median(enqueue),
            "synchronized_batch_per_call_us": (time.perf_counter_ns() - begin) / 1000 / samples}


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
    results = {"kind": kind, "shape": [rows, cols], "dtype": str(dtype), "factor": factor, "axis": axis,
               "launch": dict(actual), "candidate_hash": candidate.hash,
               "bundle_baseline_hash": candidate.baseline.hash,
               "backend_report": candidate.block_batching, "rounds": [], "seed": 42,
               "torch": torch.__version__, "torch_npu": torch_npu.__version__,
               "triton": triton.__version__, "triton_path": triton.__file__,
               "device": torch.npu.get_device_name(0), "taskqueue": knobs.ascend.enable_taskqueue,
               "fixture_sha256": hashlib.sha256(Path(__file__).with_name("block_batching_kernels.py").read_bytes()).hexdigest(),
               "samples_per_variant_per_round": samples,
               "bundle_baseline_ttir_sha256": hashlib.sha256(candidate.baseline.asm["ttir"].encode()).hexdigest()}
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
                results["baseline_ttir_identical"] = (results["standalone_baseline_ttir_sha256"] ==
                                                     results["bundle_baseline_ttir_sha256"])
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
        result = run(kind, args.rows, args.cols, getattr(torch, args.dtype), args.factor, args.axis,
                     destination, args.rounds, args.samples)
        print(json.dumps({"kind": kind, "median_speedup": result["median_speedup"],
                          "round_speedups": [r["speedup"] for r in result["rounds"]]}), flush=True)


if __name__ == "__main__":
    main()
