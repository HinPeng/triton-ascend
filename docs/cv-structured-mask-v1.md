# Structured-mask CV program batching (initial experiment)

This note describes the initial static-lane experiment. See the maintained
[CV block-batching contract](ascend/cv_block_batching.md) for current spatial,
masked batch-matmul, unified guard and AutoBlockify support.

The historical implementation version 21 used a separate CV opt-in. Current
callers use `block_batching_factor=2, block_batching_axis=0` with the original
grid; dot-containing kernels automatically select CV batching.

For residual structured memory masks, `static_masked_lanes_v1` clones the
normalized straight-line body F times within one program. Clone l uses
`original_pid = group * F + l`. All matrix operands remain rank two; original
addresses, masks, zero padding, dot precision and reductions are preserved.
This is program/grid batching, not a rank-three batch-matmul transformation.
The existing unmasked rank-three route is unchanged.

Eligibility is structural, independent of kernel names: masks must be owned
coordinate/program/scalar comparison expressions and conjunctions. Unknown or
data-loaded predicates are rejected; masked loads require explicit zero other.
The runtime proof still checks original-grid bounds, pointer arithmetic,
integer overflow, aliasing and disjoint writes. Full program groups are
required; nondivisible grids select the complete baseline before submission.
The original control-flow, shared-operand and dot-type gates remain in force.
An 8192-operation expanded-body budget bounds static code growth. Compiler
resource rejection uses the existing pre-launch fallback; execution failures
are never hidden by a baseline retry.

Correctness argument: the mapping `(group,l) -> group*F+l` is bijective for a
full grid. Each clone has the identical scalar arguments and original program
coordinates, hence identical masked accesses and arithmetic. The original
memory guard excludes cross-program read/write conflicts, so serializing these
independent computations preserves results. Source integer bounds are proved
before truncating the widened mapping to the original program-id type.

Numerical validation is additionally required for the downstream compiler.
Successful IR rewriting or device compilation alone does not prove correctness
or a performance gain. Masked matrix tails and actual applied-factor metadata
must be checked on device.
