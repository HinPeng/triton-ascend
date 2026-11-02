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

## Device validation still pending

The first Liger device case was refused by the physical-card-6 occupancy gate
before the test interpreter started. Subsequent snapshots continued to show
external processes. No new NPU correctness or timing result is claimed for this
integrated revision. The validation driver exited; it did not leave a test
process running or change the device reservation.

Before merging, rerun the integrated compiler on the six unmodified Liger
neighborhood-attention kernels: M/N concatenation with AutoBlockify, independent
BH batch matmul, admitted masked tails, non-divisible-grid fallback, and rejected
K-tail fallback. Use the original grid and verify `APPLIED` versus baseline
fallback explicitly. Include grids with more grouped programs than physical
cores and compare against independent numerical references.

The [earlier performance report](cv_auto_blockify_validation_20261102.md) retains
the completed six-kernel measurements and 172 numerical checks for the preceding
CV revision. Those results support the original transformation, but do not replace
device validation after integration with the newer unified effect plan.
