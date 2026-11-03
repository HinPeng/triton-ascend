# CV spatial concatenation: large-shape validation, 2026-10-31

Implementation: `6226cf993017595f87b558e59c175316b5fd7efa`, on
`feat/cv-block-batching-manual-v1`. This is manual factor/axis selection.
The compiler proves shared-operand M/N concatenation from IR, without kernel
names, shape dispatch tables, or changes to the packaged BishengIR backend.

All 24 scoped device cases passed the final audit: 12 performance cases,
6 masked-tail cases, and 6 whole-baseline fallback cases. The 10 initial
performance winners also passed independent-seed replay; each of their four
paired rounds exceeded 1.05x. Native compiler regressions: **759 passed**.

## Method

The six original Liger neighborhood-attention kernels used their original
32x32 output tiles and grids. Baseline and candidate used the same installed
wheel and identical inputs; candidates added only CV compiler options.
M4 means `block_batching_axis=1, block_batching_factor=4`; N4 means axis 2,
factor 4 for these kernels. The caller passed the original, undivided grid.

Both shapes used FP32 with IEEE dot input precision on Ascend950PR, physical
card 6. Independent dense random inputs and dynamic QK masks were checked
against CPU float64 equations with `rtol=atol=0.005`. Checks also covered output
canaries, unchanged inputs, actual APPLIED/factor/grid metadata, and IR with
one enlarged rank-two dot. K extent and order were unchanged.

Each adjacent baseline/candidate pair had 15 attributed device-duration samples.
Two rounds reversed execution order; winners then ran two more rounds with a
different random seed. Each round used the sample median. Times below are the
median of the paired-round medians; the two non-winners retain their initial
two-round measurements. Only complete, uncontended runs were accepted.

- `matrix128`: B=8, H=16, S=128, D=64.
- `matrix256`: B=4, H=8, S=256, D=128.
- `tails`: B=2, H=4, S=113, D=61; M4 and N2 exercise structured M/N/K tails.
- `fallback`: B=2, H=4, S=65, D=61; M2 selects the entire original baseline
  because the grid axis is not divisible by the factor.

## Results

Kernel names below omit the common `_fused_neighborhood_attention_` prefix
and `_kernel` suffix. `grad_qk` computes dQ in this kernel family.

| Kernel | Shape | Candidate | Baseline (us) | Candidate (us) | Speedup | Four-round gain |
|---|---|---|---:|---:|---:|---|
| qk (fwd) | matrix128 | M4 | 140.458 | 95.662 | 1.468x | Yes |
| qk (fwd) | matrix256 | N4 | 140.913 | 103.365 | 1.363x | Yes |
| av (fwd) | matrix128 | M4 | 72.329 | 49.991 | 1.447x | Yes |
| av (fwd) | matrix256 | M4 | 72.887 | 68.258 | 1.068x | Yes |
| grad_qk (bwd) | matrix128 | M4 | 73.020 | 54.532 | 1.339x | Yes |
| grad_qk (bwd) | matrix256 | M4 | 73.849 | 73.214 | 1.009x | No; not replayed |
| grad_k (bwd) | matrix128 | M4 | 68.805 | 53.034 | 1.297x | Yes |
| grad_k (bwd) | matrix256 | M4 | 69.458 | 71.576 | 0.970x | No; not replayed |
| grad_v (bwd) | matrix128 | M4 | 72.301 | 48.939 | 1.477x | Yes |
| grad_v (bwd) | matrix256 | M4 | 72.772 | 68.358 | 1.065x | Yes |
| grad_attn (bwd) | matrix128 | N4 | 140.204 | 89.148 | 1.573x | Yes |
| grad_attn (bwd) | matrix256 | N4 | 140.380 | 91.151 | 1.540x | Yes |

The six kernels all have a confirmed gain at matrix128. At matrix256, dQ
has no meaningful gain and dK slows down; keep baseline for those configurations.
Factor 2 did not provide a recommended gain in this sweep. Factor 4 is therefore
not a universal default, and this implementation does not select candidates
or fall back based on measured performance automatically.

## Scope and evidence

This validates single straight-line rank-two dots with a shared operand,
zero accumulator, proved coordinates, structured memory masks, and complete
program groups. It does not validate arbitrary CV control flow, multiple dots,
all shapes/dtypes, or performance of the independent A/B batching path.
The device fallback cases exercise grid divisibility; they do not claim device
coverage of every UB/L1/L0 resource-limit failure. See
[the proof and admission rules](cv_block_batching.md).

Experiment evidence is retained under `cv-spatial-merge-v1-20261031`:
`REPORT.zh.md`, all-candidate `performance.csv`, `final/performance.csv`,
`final/summary.json`, numerical/IR/launch records, profiler traces, and rejected
attempts. A compact evidence archive was copied locally and all 824 archived
evidence-file hashes were verified. Interfered runs, including a run whose
internal result completed but whose final occupancy gate failed, were excluded.

Build/runtime environments were isolated and registered separately. Both final
snapshots passed scoped registry, package-prefix, hash, and dependency checks.
The global registry check stopped at a pre-existing failed snapshot belonging
to the old `ta-run-liger-baseline-20261028` environment; that history was preserved.
The test controller and its cleanup guard exited, and task/card locks were
released after validation.

- Wheel SHA256: `0acac5865c9b2f1fbc59b6ca8daa103815ef545da3210c8705369894fbc7fe47`.
- Installed libtriton SHA256: `77b9a4d0506d2f99436afc8d1bf5fbaf55422dd69bbda29621621296ffe1e6a8`.
- Compact evidence archive SHA256: `59abfba106af8b9228c40d7e5fd686e1455d8ae584910636dfbd3f0020b7ba19`.
