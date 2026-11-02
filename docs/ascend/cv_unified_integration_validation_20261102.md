# CV integration with the unified effect plan

The integration is based on `feat/block-batching-ta-only` at
`be252d51c0224e07a4aa208a5b7e414f3de83dd6`. Executable code and tests were built
and checked at `c06b1a4df3d710c6dba23f696da577e499a7feb9`.

## Completed validation

| Check | Result |
| --- | --- |
| Full native TA wheel build in an independent build environment | Passed |
| Installed-wheel import paths and payload hashes in a separate runtime | Passed |
| All `test_block_batching*.py` host tests, excluding `_e2e` modules | 1004 passed, 3 deprecation warnings |
| `prefix_zero_load.mlir` through installed `triton-opt`, `--verify-each`, and FileCheck | Passed |
| pre-commit on changed files, including yapf, clang-format and mypy | Passed |
| Physical-card-6 device regression on six unmodified Liger CV kernels | 19 cases, 92 checks passed |

The added integration tests exercise v3 guards for M/N on all three program
axes; masked-batch padding and reduction obligations; alias/effect rejection;
source/plan fingerprint binding; and transactional refusal of opaque device
addresses in CV. Existing VV unified-plan tests are included in the same run.

The wheel SHA256 is
`7d030ab4784a25f45b1160269b6cad1ca41ad1dab8a10947777f552eceedc99e`;
the installed `libtriton.so` SHA256 is
`5840f711f87bef983c074ab44242b7ae74a5b5ca751a3876ae3429be085f547d`.
The packaged BiShengIR consumer remains `8995ba08834c`, with unchanged payload
hashes. AscendNPU-IR is `bb13ef6120526aa3df6f950006af5ec0e585d2f6`; CANN is 9.1.0.

## Device regression completed

The installed wheel was exercised on physical card 6, Ascend950PR. All 19 cases
passed with two input seeds (`20261102`, `20261103`), FP32 inputs and IEEE dot
precision. Each execution was compared with an independent CPU float64 reference
at `rtol=atol=0.005`; candidates were also compared with the corresponding
baseline. Output canaries and input immutability passed. The maximum absolute
error across the recorded checks was `3.0517578125e-05`.

The six kernels are `_fused_neighborhood_attention_qk_kernel`,
`_fused_neighborhood_attention_av_kernel`,
`_fused_neighborhood_attention_grad_qk_kernel`,
`_fused_neighborhood_attention_grad_k_kernel`,
`_fused_neighborhood_attention_grad_v_kernel`, and
`_fused_neighborhood_attention_grad_attn_kernel` (LK07 through LK12).

| Scenario | Shape `[B,H,S,D]`, FP32 | Kernels / factors | Cases | Checks |
| --- | --- | --- | --- | --- |
| Large M/N with AutoBlockify | `[8,16,128,64]` | LK07-LK12; baseline, M2, N2 | 6 | 36 |
| M/N masked tails | `[2,4,113,61]` | LK07/LK08; baseline, M4, N2 | 2 | 12 |
| Non-divisible grid fallback | `[2,4,65,61]` | LK07; baseline, requested M2 | 1 | 4 |
| Small independent BH batch | `[2,4,32,32]` | LK07-LK12; baseline, BH2 | 6 | 24 |
| BH admitted output tails | LK07 `[2,4,29,32]`; LK08 `[2,4,32,29]` | baseline, BH2 | 2 | 8 |
| BH reduction-tail fallback | `[2,4,32,29]` | LK07; baseline, requested BH2 | 1 | 4 |
| BH AutoBlockify loop | `[8,16,32,32]` | LK07; baseline, BH2 | 1 | 4 |
| **Total** | | | **19** | **92** |

These comprise **38 baseline**, **50 actually applied merged executions**, and
**4 expected fallback executions**. The fallback reasons were
`CV_GRID_NOT_DIVISIBLE` and `CV_BATCH_REDUCTION_MASK_NOT_PROVEN`, each checked
with both seeds. No missing record or fallback is counted as an applied merge.

Every applied candidate checked the requested factor, original/grouped grid,
v3 guard and source/plan fingerprint binding. AutoBlockify and the launcher's
`AUTO_MAP` flag were enabled for every artifact. The large M/N cases and BH loop
case checked that grouped programs exceeded the physical-block count. The BH
loop has 64 grouped programs scheduled on 28 physical blocks.

All 18 applied BH checks also verified the typed `hivm.hir.batchMmadL1` operation,
its factor-sized batch dimension, the backend option and the corresponding
executed ELF hash. A final ELF substring is not used as a lowering contract:
this backend emits Catlass/Mmad symbols rather than retaining `batch_mma`.

The resumable run retained three interrupted attempts: an initial occupancy-gate
refusal, missing immutable small-case provenance files, and an overly specific
ELF-symbol assertion. The fixture files were restored from the previously
validated package and checked against their frozen hashes. The assertion was
replaced with the typed-operation proof, with negative checks for ordinary
matmul and a mismatched factor. No compiler, kernel, wheel or installed package
changed during the device regressions. Completed cases were retained.

Every case passed the pre-run occupancy and health gate. Initially the gate
admitted one exited process record only with a fresh, task-bound host absence
witness. Two post-run snapshots retained that record; another briefly showed an
unclassified PID, which disappeared before the next case. Those observations
are preserved and do not establish timing isolation. **This run reports
correctness only and adds no performance measurements.**

The controller and cleanup/witness helpers have stopped, task/card locks are
released, and the final card snapshot is healthy and empty. The runtime is
registered as `validated-scoped`; build/runtime snapshots and their manifests
were verified.

## Evidence

Remote evidence root:
`/home/z00977770/isolated-ta/builds/cv-pr-integration-20261102`.
`device-regression-summary.json` audits all 19 result records, expected variants,
both seeds, payload identities, lowering evidence and binary hashes.

The archive `device-regression-evidence-20261102T093713Z.tar.gz` contains the
resumable scripts, frozen fixtures, logs, IR/ELF artifacts, failed-attempt
diagnostics, occupancy records, environment snapshots and a per-file manifest.
Its SHA256 is
`befd3218b42d29555ee742a41d5225f8ee96a0d34aaf4b1e665ba79b64a8804c`.

The [earlier performance report](cv_auto_blockify_validation_20261102.md) retains
the completed six-kernel measurements and 172 numerical checks for the preceding
CV revision. Those results support the original transformation, but do not replace
the scoped correctness regression above after integration with the newer unified
effect plan. Other shapes, dtypes, control flow and performance remain outside
this integration regression.
