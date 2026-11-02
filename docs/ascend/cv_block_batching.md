# Manual CV block batching: independent lanes and spatial concatenation

This opt-in extension integrates with the unified block-batching effect plan.
CV does not choose a factor automatically.

```python
compiled = kernel[original_grid](
    *args,
    block_batching_cv=True,
    block_batching_axis=1,  # a proven M-tile axis; use the kernel's actual axis
    block_batching_factor=4,
)
print(compiled.block_batching)
print(compiled.last_launch)
```

Pass the **original grid**, never a pre-divided grid. The existing guarded bundle
owns the grid conversion, three original-grid hidden arguments, binary, workspace,
and launch metadata as one complete artifact. Factor 1 compiles the original
kernel. Valid CV factors are powers of two from 2 to 64; backend resource admission
may reject any of them. The batch-matmul option selects the existing backend path;
it is not a new automatic selection mechanism.

## Semantics and supported domain

The compiler chooses a lowering from IR dependence on the requested axis;
there are no Liger names, shape tables or host-side tile overrides in the pass.
Floating inputs may be FP16, BF16 or FP32 with FP32 accumulation. Device validation
must be reported separately for the actual precision/shape tested. The original
memory-domain, integer-overflow, alias and exclusive-store guards remain required.
CV uses the same v3 source/plan fingerprint as VV. The independent masked-batch
path additionally checks its original-coordinate padding and reduction
obligations. VV's terminal-read relaxation for opaque device addresses does not
authorize a CV matrix transformation: those addresses are rejected for CV.

### Shared B: concatenate M; shared A: concatenate N

For a single straight-line rank-two dot, the new `spatial_concat_v1` path proves
one canonical coordinate `pid(axis) * T + arange(0, T)`. All uses of that program
ID must enter through this coordinate. It rewrites the coordinate to
`group * (F*T) + arange(0, F*T)` and propagates the expanded tensor dimension
through addresses, loads, structured masks, transpose, dot and elementwise
operations. `spatial_m_v1` requires only A's M axis to expand; `spatial_n_v1`
requires only B's N axis. The shared operand and K extent/order are unchanged.
The initial accumulator must be constant zero.

The proof is ordinary block matrix multiplication:

```
concat_rows(A0, A1) @ B = concat_rows(A0 @ B, A1 @ B)
A @ concat_cols(B0, B1) = concat_cols(A @ B0, A @ B1)
```

Output stores cover the same disjoint coordinates as the original F programs.
A shared `arange` definition is never resized globally: it may also feed K or
the unchanged matrix. Uniform tensors are resized per use only when proved
invariant along the expanded dimension (splat, singleton broadcast, or proven
pointwise/transpose compositions). Independence from `pid` alone is insufficient.
For example, an epilogue bias indexed by the original *local* M coordinate is
currently rejected, rather than silently interpreted as a larger global bias.

Boundary memory masks retain their exact expressions and zero load alternatives.
A loaded data mask in an elementwise epilogue, such as the QK neighborhood select,
is retained. Data-dependent memory predicates remain outside the admitted domain.
The first spatial implementation rejects scalar uses escaping the coordinate,
K-axis expansion, multiple dots, unproved reshape/reduction/scan, nonzero Acc,
and stores that do not concatenate. Rejection leaves the complete input IR intact.

### Independent matrix pairs

With `enable_hivm_batch_matmul=True`, `independent_batch_v1` stacks independent
matrix pairs into `[F,M,K] @ [F,K,N] -> [F,M,N]`. The added batch dimension never
becomes part of K, so each output lane depends only on its own input pair.
`rank3_masked_batch_v1` retains rectangular, zero-padded load masks and the
original output masks. It does not require every memory mask to be true.

This path first zero-initializes the complete load destination, then writes each
lane's valid GM data directly into its destination subview. This avoids an
intermediate L1-to-L1 copy, which the packaged backend cannot lower. Transposed
operands are loaded as a contiguous batch and then transposed as a whole.

The generated dot carries an internal `tt.cv_batch_full_mkn` marker. Lowering
passes its complete logical M/K/N tensor extents directly to `batchMmadL1`, with
`already_set_real_mkn`. The packaged consumer otherwise tries to infer those
extents from nested subview writes and can leave its zero placeholders intact.
These dimensions describe the mathematical, padded tensors; they do not assert
that a GM access mask is true. Input precision and accumulation semantics are
preserved. Unmarked rank-three dots keep their existing lowering.

Admission remains conservative: K must be fully present, and both physical
operand axes must retain the same final 32-byte padding extent as the complete
tile (quantum 8 for FP32, 16 for FP16/BF16). Empty or more distant tails select
the full baseline before launch. This is a backend-layout restriction in addition
to ordinary bounds and alias checks; arbitrary zero-padded K tails are not yet
supported. Numerical validation is required for each admitted layout and dtype.
If JIT specialization makes a prefix bound constant, a proven violation of these
same limits is rejected before calling the backend. This prevents a statically
ineligible `1x1x1` candidate from reaching unsupported ND2NZ lowering before its
runtime guard can run. Dynamic arguments and grid values are never guessed by
this early check; their existing launch checks remain mandatory.

Without the batch-matmul option, structured masked memory still uses
`static_masked_lanes_v1`, cloning the rank-two body F times. That path must not be
reported as a hardware batch-matmul optimization.

### Common limits

CV requires `original_grid[axis] % F == 0`. A partial **group of programs** selects
the entire baseline. An original program's partial M/N/K tile can still qualify
for the spatial path when its original structured masks and memory guard prove
it safe. These are distinct conditions; no workload is rounded or omitted.

The masked independent-batch and shared-operand M/N spatial paths can use backend AutoBlockify after the
guarded CV grid transform. The launcher first maps the original grid to grouped
programs, then caps physical blocks; the backend loop visits every grouped
program at a stride of the physical core count. The original-grid hidden
arguments remain unchanged. This combination is admitted only for MIX mode,
`rank3_masked_batch_v1`, `spatial_m_v1` or `spatial_n_v1`, and an empty
compiler-derived AutoBlockify blacklist.
Other CV lowerings retain their prior AutoBlockify exclusion. Correctness and
performance still require a device test with more grouped programs than physical
blocks; enabling the flag on a smaller grid does not exercise the loop.

Only straight-line normalized IR is accepted. A statically unrolled source loop
can qualify if no control-flow region remains. Dynamic loops/conditionals,
dot-scaled, pre-batched rank-three dot, atomics and cross-program synchronization
remain rejected. VV memory-view folding is disabled for CV. CV owns logical
grouping; AutoBlockify may schedule those grouped programs only in the admitted
masked independent-batch and spatial paths above.

## Admission and failure

The baseline compiles independently first. Before accepting a CV binary, validate
its hash, matching native implementation, MIX/AIC mode, exact v3 grid contract,
workspace, task type, absence of conflicting mappings and global lock use.
Normal backend compilation admits resource usage. Recognized UB, L1/cbuf and
L0A/B/C (`ca`, `cb`, `cc`) overflow selects baseline before submission, preserving
the diagnostic. There is no runtime replay after device execution starts.
Unknown internal compiler errors remain errors and must be investigated.

Compilation success or baseline fallback is not optimization evidence. An
accepted measurement requires `last_launch.runtime_status == 'APPLIED'`, the
requested factor, numerical comparison and uncontended device timing. Compare
baseline and candidate in adjacent interleaved rounds on identical inputs.

The packaged BishengIR backend is unchanged; this revision changes TA's rewrite
and lowering into its existing operations. Tests and performance results are
scoped to the registered environment and must not imply all CV kernels work.

The [2026-10-31 large-shape validation](cv_spatial_validation_20261031.md) covers
six unmodified Liger kernels, masked tails and full-baseline grid fallback.
Ten of twelve performance configurations retained a gain across four paired
rounds; the two non-winning configurations are reported explicitly.

The [2026-11-02 AutoBlockify validation](cv_auto_blockify_validation_20261102.md)
records the later six-kernel experiment with AutoBlockify enabled on both sides.
Those timings belong to the recorded pre-integration commit. The
[unified-plan integration validation](cv_unified_integration_validation_20261102.md)
records the new native build, 1004 host-test passes, and 19 device cases with
92 numerical checks for M/N, independent BH, admitted tails, fallback and
AutoBlockify scheduling. It adds no new performance measurements.
