# Latest-main Liger CV validation (2026-11-05)

This is the user-authorized latest-main comparison of the existing 12 large Liger CV specifications. It does not replace historical frozen-main records or claim validation of all FLA/control-flow cases.

## Revisions and method

- Main: `b3c899b8de4a1061d93f710262ef46e61e4ab4a0`.
- CV wheel source: `0b08285ec30f3f7ff14fcbaabb14d98a768977fe`; subsequent commits `be3667a665` and `437fae3829` only repair host-test fixtures.
- NPUIR gitlink: `4eb12142e16fc8c7ed33c3b568658b773ae0d1c9`; actual official binary source: `d51bdc6ca9cf0b10488eb47a52793607cfc1d99d`.
- Official workflow/artifact: `41804999588` / `13312315353`, x86_64; ZIP SHA256 `66b40fe0552107f5cb56a378b28d6a78225a979c2268fbe7e35e6504f802a6b5`.
- Separate build/runtime environments for main and CV; identical CANN 9.1.0, dependencies and NPUIR. Physical card 6, Ascend950PR, serial `10265A283915`, PCI `0000:57:00.0`.
- Liger source: `781a8aef64bf6105bdefb5f4c991e0d14c08e793`; six unchanged neighborhood-attention kernels; FP32, IEEE dot; M/N tile 32. Inputs `[B,H,S,D]` are `[8,16,128,64]` and `[4,8,256,128]`.
- Main is independently compiled, executed and checked. The candidate process replays the exact main binary after verifying all artifact hashes, source and launch parameters. A disabled CV compiler is not used as the main baseline.
- CPU float64 reference; rtol=atol=0.005; input immutability and output canaries; two seeds, two rounds each. Each round contains 15 neighboring, alternating main/CV sample pairs. Report full attributed Triton kernel device duration.
- Stable improvement requires every one of four round speedups to exceed 1.05x. Overall ratio uses the median of four round medians for each side. Winner selection is limited to M2/M4/N2/N4, not a global tile search.

## Results

All 12 specifications passed. All 44 eligible candidates were actually applied, numerically correct and above 1.05x in all four rounds; no final fallback or compile/accuracy failure. All candidates used spatial M/N concatenation with AutoBlockify enabled. Evidence includes 720 independent-main samples and 5,280 paired device samples.

| Kernel | B/H/S/D | Best | Paired main (us) | CV (us) | Speedup | Latency reduction |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| qk (fwd) | 8/16/128/64 | M4 | 26.355 | 16.377 | 1.609x | 37.9% |
| qk (fwd) | 4/8/256/128 | M4 | 43.404 | 28.279 | 1.535x | 34.8% |
| av (fwd) | 8/16/128/64 | M4 | 23.343 | 16.078 | 1.452x | 31.1% |
| av (fwd) | 4/8/256/128 | M4 | 40.663 | 27.583 | 1.474x | 32.2% |
| grad_qk (bwd) | 8/16/128/64 | M4 | 24.523 | 17.107 | 1.434x | 30.2% |
| grad_qk (bwd) | 4/8/256/128 | M4 | 41.853 | 28.227 | 1.483x | 32.6% |
| grad_k (bwd) | 8/16/128/64 | M4 | 24.678 | 17.214 | 1.434x | 30.2% |
| grad_k (bwd) | 4/8/256/128 | M4 | 42.151 | 28.614 | 1.473x | 32.1% |
| grad_v (bwd) | 8/16/128/64 | M4 | 23.493 | 15.768 | 1.490x | 32.9% |
| grad_v (bwd) | 4/8/256/128 | M4 | 40.490 | 27.617 | 1.466x | 31.8% |
| grad_attn (bwd) | 8/16/128/64 | N4 | 25.838 | 15.570 | 1.659x | 39.7% |
| grad_attn (bwd) | 4/8/256/128 | M4 | 42.966 | 26.959 | 1.594x | 37.3% |

For output D=64, grid_N=2, so N4 is excluded by divisibility (four exclusions, 44 eligible candidates). Negative or absent results would remain explicit; no slower candidate is overwritten by the baseline.

## Host regression and limitations

Host regression: 1,112 collected, 1,082 passed, 28 skipped by existing Ascend950 restrictions, and two pre-existing CV test expectation failures. No new failure remains after review. The two failures are the varying-bound diagnostic expectation and the formerly-rejected loop-write expectation in `test_unknown_or_unsupported_regions_keep_complete_baseline`; they are not counted as passes.

The main host compiler-cache test fixture linked Conda C++ libraries without GNU build IDs. Both unchanged main and CV correctly fell back to live version queries, contrary to the test expectation. A C-only driver fixture removes that unintended dependency; all 20 targeted tests pass on both installed wheels. Compiler cache identity checks are unchanged. Separate compiler-command fixtures now mock the capability query for their fake compiler path.

One initial NPU attempt was interrupted by occupancy/query errors and excluded; its trace and result are preserved. Final accepted runs have successful before/after card gates and clean per-round telemetry. Cleanup used only identity-verified foreign card-6 process termination; no device reset or container/scheduler stop.

Validation covers these straight-line, grid-divisible large-shape spatial candidates only; it is not additional validation of batch-mode tails, uniform-loop fusion, all dtypes or all devices. The official NPUIR was built with CANN 9.2.0-beta.2; the recorded scope executed successfully with CANN 9.1.0.

## Evidence

Local task: `outputs/cv-main-sync-20261105/`. Docker task: `/home/z00977770/isolated-ta/builds/cv-main-sync-20261105` via `triton-ascend-37`. `PROTOCOL.zh.md`, `scope.json`, `summary.json`, `results.csv`, `paired-samples.csv`, `native-reviewed.json`, per-run preflights, IR, binaries, profiler traces and telemetry describe the exact scope. Environment registry and snapshots are under `/home/z00977770/env`.
