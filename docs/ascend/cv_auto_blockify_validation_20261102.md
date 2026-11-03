# CV M/N concatenation with AutoBlockify: device validation

This experiment used compiler commit
`0fc8fc7a12e7c1ac2c9fd1fc4c8217d74d0f51e1`, before integration with the
unified effect proofs at `be252d51`. It is historical device evidence for the
CV transformation, not a claim that a later integrated commit was measured.

## Method

- Six unmodified Liger neighborhood-attention kernels, FP32 with IEEE dot.
- Shape A: `[B,H,S,D]=[8,16,128,64]`; shape B: `[4,8,256,128]`.
- Original output tile: 32 by 32. K tile: the complete logical reduction
  dimension rounded to a power of two, as in the earlier large-shape experiment.
- M2/M4 merge two/four programs along axis 1; N2/N4 along axis 2. The caller
  passes the original grid. The compiler alone expands the matrix tile.
- Both baseline and candidate use AutoBlockify. The candidate additionally
  uses M/N concatenation. Speedup does not isolate AutoBlockify's contribution.
- Two independent input seeds, two adjacent interleaved timing rounds per
  seed with reversed order, fifteen device samples per variant in each round.
  Each reported duration is the median of four per-round medians. All four
  individual speedups must exceed 1.05 to qualify as a stable gain.
- CPU float64 reference with `rtol=atol=0.005`, unchanged inputs and output
  canaries. Candidate status must be `APPLIED`, with the requested factor,
  expanded rank-two dot, grouped grid and actual AutoBlockify scheduling.
- Hardware: physical card 6, Ascend950PR; CANN 9.1.0; packaged BiShengIR
  `8995ba08834c`; AscendNPU-IR `bb13ef6120526aa3df6f950006af5ec0e585d2f6`.
  Separate build/runtime environments and fresh per-run compiler caches.

## Fastest measured candidates

Kernel names below omit the common `_fused_neighborhood_attention_` prefix
and `_kernel` suffix. Durations are complete Triton kernel device time in us.

| Kernel | Shape | Candidate | Baseline us | Candidate us | Speedup |
| --- | --- | --- | ---: | ---: | ---: |
| qk (fwd) | A | M4 | 140.786 | 38.942 | 3.615x |
| qk (fwd) | B | M4 | 141.090 | 58.459 | 2.414x |
| av (fwd) | A | M4 | 72.283 | 31.137 | 2.321x |
| av (fwd) | B | N4 | 72.941 | 51.153 | 1.426x |
| grad_qk (bwd) | A | M4 | 73.244 | 32.355 | 2.264x |
| grad_qk (bwd) | B | N4 | 73.968 | 51.731 | 1.430x |
| grad_k (bwd) | A | M4 | 68.886 | 31.021 | 2.221x |
| grad_k (bwd) | B | N4 | 69.624 | 49.848 | 1.397x |
| grad_v (bwd) | A | M4 | 72.349 | 30.904 | 2.341x |
| grad_v (bwd) | B | M4 | 73.035 | 50.781 | 1.438x |
| grad_attn (bwd) | A | N4 | 140.011 | 38.252 | 3.660x |
| grad_attn (bwd) | B | M4 | 140.212 | 57.169 | 2.453x |

All 44 measured M2/M4/N2/N4 candidates met the four-round gain criterion.
These are enumerated measurements; the compiler does not automatically choose
the fastest factor. A candidate slower than baseline would remain in the raw
data rather than be replaced by the baseline number.

## Correctness and compiled execution

The suite completed 24 cases: twelve performance cases, six masked-tail cases
at `[2,4,113,61]` (M4/N2), and six non-divisible-grid cases at `[2,4,65,61]`
(M2, complete baseline fallback). All 86 configurations passed both seeds,
for 172 correctness checks. The host suite passed 861 tests.

The twelve selected performance binaries and two masked-tail binaries were
re-lowered with diagnostic output. Their hashes matched the executed binaries;
LLVM IR contained stride-28 loops, and the profiler recorded 28 physical AIC
blocks despite hundreds of grouped logical programs.

One initial grad_v shape-B attempt stopped before timing on a device health
warning (`UB_PORT`, `81B18603`). Its evidence was retained and excluded. After
the card recovered and was idle, only unfinished cases resumed. Retained
timing-round telemetry observed no foreign process.

The compact evidence archive SHA256 is
`9eefbef535c5d40f586919dc50996dff0e12b1b4a6855b51e7225c859f74eb38`.
The full candidate/round/correctness tables and binary evidence are retained
with the isolated `cv-spatial-auto-blockify-v1-20261102` experiment. The result
does not extend validation to dynamic control flow, arbitrary multi-dot
kernels, other dtypes, or other hardware.
