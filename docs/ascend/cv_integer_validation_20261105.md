# INT8 CV validation, 2026-11-05

The CV rewrite now admits `i8 x i8 -> i32`, preserving each original K
reduction, integer widths, masks and accumulation order. Shared-operand M/N
concatenation and lane-wise packed-byte operations passed device validation.
The unchanged Liger `matmul_kernel` still selects its complete baseline:
its nested pointer recurrence is not proved. No Liger merge speedup is claimed.
The user deferred further address-proof development after this experiment.

## Revisions and environment

- Main: `b3c899b8de4a1061d93f710262ef46e61e4ab4a0`.
- Tested candidate wheel: `a25e2dbc8bfaee6d4725767a24139ec378e31b0a`;
  implementation version 37. Later test/documentation commits do not change
  the tested compiler binary.
- Candidate wheel SHA-256:
  `cace7865bef47730be0e38926666c0a19a84fdf30aa93f4fb1804a6cab31cf44`.
- Installed candidate libtriton SHA-256:
  `65821223946192d85d1448c12e4a0ef1857bf7e0c95a9ec93634574ebfa93ee5`.
- Matching official NPUIR payload on both sides:
  `d51bdc6ca9cf0b10488eb47a52793607cfc1d99d`, workflow `41804999588`,
  artifact `13312315353`. Source gitlink
  `4eb12142e16fc8c7ed33c3b568658b773ae0d1c9` is recorded separately.
- Explicit CANN 9.1.0, physical card 6, serial `10265A283915`,
  PCI `0000:57:00.0`. Separate build/runtime environments and fresh run caches;
  task/card locks, import/binary audits and occupancy checks were retained.
- Docker evidence root:
  `/home/z00977770/isolated-ta/builds/cv-i8-v2-20261105`.

## Device checks

All 18 standalone checks passed with full-range signed INT8 inputs and an
independent CPU double-precision exact integer reference (`rtol=atol=0`):

| Path | M/N/K | Original output tail | Factors | Applied F2/F4 |
| --- | --- | --- | --- | --- |
| Shared B, concatenate M | 128/32/64 and 127/32/64 | full / M tail | 1, 2, 4 | 4/4 |
| Shared A, concatenate N | 32/128/64 and 32/127/64 | full / N tail | 1, 2, 4 | 4/4 |
| Four independent pairs | 32/32/64 and 32/31/64 | full / N tail | 1, 2, 4 | 4/4 |

The independent-pair path is `static_integer_lanes_v1`: F independent rank-two
dots in a grouped program, not one rank-three batch MMA. The initial candidate
actually failed to link `_mlir_ciface_batch_mma_tile_int8_t_to_int32_t` with the
current official consumer. That failure is preserved under `cv-i8-20261105`;
the final candidate avoids generating that unavailable call. Spatial M/N
still produces a genuine wider rank-two dot. NPUIR itself was not changed.

A separate packed-int2 fixture with M/N/K=128/32/256, four bitplanes and two
inner iterations passed F1/F2/F4. F2/F4 applied
`spatial_m_uniform_control_v1`. It uses direct, provable coordinates; it is
not the unchanged Liger kernel. Its first frontend attempt omitted
`out_dtype=tl.int32`; this harness error and its corrected rerun are retained.

The committed device regression file separately passed all 14 tests: the 12
F2/F4 matrix/tail combinations above plus two packed nested-loop cases, with
exact results, output canaries and immutable inputs. These overlap the
standalone checks and are not 14 additional workload specifications.
No performance conclusion is drawn from these correctness-only fixtures.

## Unchanged Liger kernel

Original `experimental/mm_int8int2.py::matmul_kernel`, source SHA-256
`4eb4bd969c1e92b8a4ff0e6a13b0213090f56dcabe3a2763b5bb16f34a19e883`:
A[256,2048] INT8, Bpack[512,512] UINT8, decoded B[2048,512] in {-1,0,1,2},
C[256,512] INT32. Original tile=64/128/64, GROUP_SIZE_M=8,
warps/stages/ctas=4/4/1, grid=16. The recorded configuration was frozen
through the original autotuner's JIT function; neither side autotuned again.

| Request | Paired main, us | Candidate's actual execution, us | Main / execution | Status |
| --- | ---: | ---: | ---: | --- |
| F2 | 54.5365 | 54.4500 | 1.001589x | Baseline selected; no merge |
| F4 | 54.4410 | 54.4680 | 0.999504x | Baseline selected; no merge |

The independent main measurement was 54.1590 us. Each pair used two seeds,
two rounds per seed, and 15 attributed device samples per side per round;
four round medians were aggregated by their median. There were 60 independent
main samples and 240 paired samples. Main binaries were independently compiled
and hash-verified before paired replay. Inputs cover the full INT8 range;
independent unpacking, exact reference results, canaries and input immutability
passed. AutoBlockify remained enabled on the baseline fallback.

Both requests reject with `UNKNOWN_POINTER_ROOT`. The outer `scf.for` yields
the result of an inner loop carrying A's pointer; the existing provenance
analysis accepts a direct invariant `tt.addptr` backedge, not this nested
recurrence. Saved real IR was diagnosed without mutation. Dynamic grouped PID
division/remainder, `min` and modulo addresses also remain outside the new
type support; resolving the first rejection would not prove the entire kernel.

## Native regression and limits

After restoring missing, exact-version test fixtures and rerunning only the
five affected registry tests: **1105 passed, 28 skipped, 2 known parent
expectation failures**. The latter are the existing VV control-flow cases
expecting `VARYING_LOOP_BOUND` and `LOOP_WRITE_NOT_IMPLEMENTED`; they match the
parent's saved failure list. The initial collection and missing-fixture errors
are retained separately and are not reported as compiler successes.

The 23 new native checks cover integer M/N types, preserved K, byte unpacking,
zero accumulators, unsupported types/overflow attributes, rank-two independent
lanes and strict report contracts. The change does not enable arbitrary byte
operations, integer dot types, unproved addresses, partial program groups,
or a native integer batch-MMA implementation. Tested tiles do not establish
all shapes, K-overflow behavior or grid-stride AutoBlockify scheduling.

Reproduction uses the registered `run.sh candidate 6` entry with
`validate_integer_device.py`, `validate_packed_loop.py`, or
`native_tests.py test_block_batching_cv_integer_e2e.py`.
Native-only checks use `run.sh candidate host native_tests.py`.
Raw runs, preflights, IR, traces, samples, failed attempts and audited summaries
are retained in the evidence root; `STATUS.zh.md` is the Chinese result record.
