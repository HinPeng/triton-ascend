# Automatic constexpr reuse

Set `TRITON_ASCEND_ENABLE_DYNAMIC_REUSE=1` before starting the process. The
switch is read once. Kernel definitions and ordinary `kernel[grid](...)` calls
do not change. Without the switch the optional backend handler is absent.

Foreground scalar dynamicization (D) is available on NPU targets in any
compile mode accepted by the backend, without requiring specific CANN or
torch-npu versions. Cross-tile reuse (T) is available on NPU targets in any
backend-supported compile mode except `simt_only`, without restricting the device
model. Pure SIMT retains the requested tile while still allowing D. The execution
process must keep its device and compilation environment fixed.

The initial hardware validation used CANN 9.1.0 and
torch-npu 2.13.0.dev20260930; the validation report records that tested environment.

## Supported foreground behavior

* Independent AST/use-site analysis classifies every constexpr parameter.
  It resolves actual operation/helper identities, joins branches, and computes
  bounded loop dependency fixed points. It does not import the autotuner DSL.
* Python bool, int and float constexpr data uses (including numeric subclasses)
  without static-use constraints become unannotated runtime parameters on private
  JIT copies. The native JIT infers their types and owns any recompilation when
  the inferred signature changes with the value range. Reuse neither selects a
  fixed scalar ABI nor promises one binary across different inferred types.
  Value and alignment specialization are disabled for the rewritten slots.
  Local constexpr assignments and every affected helper call are copied too.
  Original JIT source, parameter descriptors, and device caches remain intact.
* Shape, static control, static loops, dtype/axis attributes and unknown uses
  retain their original specialization. Scalar arithmetic, conversion, promotion,
  numeric results and errors follow native Triton behavior, including narrow/unsigned
  operations, floating operands, casts, where and cdiv. There is no D-side numeric
  domain guard, host expression evaluator, fixed-width gate or divisor-sign gate.
* D adopts Triton's runtime numeric semantics. It does not guarantee identical
  results to Python constexpr precomputation: integer overflow, negative division,
  floating rounding and weak-scalar promotion can differ. Tests compare the new
  D path with native runtime behavior where these semantics matter. Structural
  eligibility alone is not a numerical-equivalence proof for every input.
* Exact READY lookup precedes family lookup. A local-axis use-site analysis
  tracks the tile separately in scheduling, tensor shapes and computed values.
  It recognizes complete traversals such as `range(0, N, B)` and
  `range(tl.cdiv(N, B))`, with `arange(0, B)` forming the logical index and a
  mask limiting accesses to `N`. The outer grid and its partition may use any
  of the three program axes, provided they are independent of the reused tile.
  Parameter names, positions, function names and the number of sequential
  passes are not part of the rule.
  Index expressions normalize associative/commutative sums and products,
  integer identities, reversed comparisons and positive integer ceil-division
  spellings. Original expressions remain attached to the proof and are checked
  for both requested and candidate tiles, so cancellation cannot hide overflow.
  Floating division/casts are not treated as equivalent integer index arithmetic;
  explicit narrow/unsigned index types require an additional arithmetic model.
  Logical axis identity is distinct from loop provenance: sequential traversals
  can share an axis while retaining their own induction source and source line.
* Tile-independent constexpr conditions select static `if`/conditional-expression
  paths. Their typed values participate in the analysis-cache context, and
  controlling parameters retain specialization. Runtime or tile-dependent
  conditions decline the local-axis proof. JIT helpers bind positional/keyword
  arguments and scalar defaults in their own scope, propagating axis information
  through nested calls, loads, returns and tuple unpacking. Helpers share the
  analysis budget; recursion and returns that truncate a tile traversal decline.
* Pointwise arithmetic, casts, broadcast and singleton reshape compose through
  the same analysis as supported accumulators. A loop-carried sum/min/max state
  must start at the operation's identity, receive neutral padding, and be
  reduced along the tile axis before it escapes to ordinary computations or
  stores. Arbitrary recurrences, nonlinear uses of unfinished accumulation,
  unmasked local accesses and tile-dependent computed values decline reuse.
  The layernorm example is a test fixture for this common analysis; it has no
  dedicated production matcher or argument layout.
* Runtime guards use the inferred access expressions and masks to bound the
  whole grid, check contiguous tensor storage, and prove distinct output
  coordinates. Dtype is checked by the native candidate signature. Overlapping
  accesses involving a store require the same byte mapping and element width.
  Local shapes for both old and requested tiles must fit Triton's tensor-size
  bound. Internal tile steps and shapes follow the reused binary, while the
  outer grid and arguments remain fixed.
* The same axis analysis recognizes direct masked tiling without a loop:
  `index = program_id(axis) * B + arange(0, B)`, with `index < N` protecting
  accesses. The requested grid on that axis must be `ceil(N / B)`. When an old
  binary uses `B_old`, dispatch changes that grid dimension to `ceil(N / B_old)`
  and binds `B_old`, keeping every other grid dimension unchanged. The selected
  grid is installed only after candidate ABI checks succeed. Tuple, list and
  callable grids are supported; callbacks are evaluated once with the requested
  arguments, and rejected candidates leave that requested grid for fallback.
  Direct uses of the changed axis's pid/program count in values or addresses,
  per-program scalar writes, and tile-wide reductions without a full-axis proof
  decline reuse. An explicit `grid_num_tiles` compiler contract prevents changing
  the grid. This rule does not impose the grid-stride vector-core limit.
* The earlier grid-stride elementwise rule also handles external tile counts,
  recomputing the count for the selected binary. It retains its vector-core
  limit; local-axis traversal does not impose that limit on the outer grid.
  Unknown operations, control flow and address mappings continue through JIT.
* Repartitioning a floating-point accumulation may change rounding, as in
  ordinary autotuning. The local-axis proof preserves the supported operation's
  mathematical coverage and dependencies, not bitwise results across tiles.
  Host tests cover unrelated pointwise/multi-pass/sum/max kernels and execute
  the layernorm fixture with NumPy. NPU numerical qualification is still required.
* A miss continues the existing JIT tail, including its native memory/disk cache.
  T-only uses the original ASTSource and native key. Rewritten sources add typed
  constants and transform/ABI versions before using the native key algorithm.
* READY registration requires complete native initialization and nonzero native
  handles. The two indexes share executable objects. Registry quotas do not
  prevent a normal foreground JIT call. Initialization failures before launch
  may fall back once; metadata/hooks/launch errors are never replayed.

Callable grids support grid-stride, local-axis and direct-grid T rules: when checking a
registered cross-tile candidate, the grid is evaluated once with the requested arguments.
Grid-stride and local-axis rules retain it; direct-grid rules install a proved replacement
only for a selected candidate. Grid-stride still requires the requested external block
count to equal `ceil(N / B_requested)` and recomputes it as `ceil(N / B_selected)`.
Fallback retains the requested grid. A miss without such a candidate evaluates it in the JIT
tail as before. Warmup, graph capture, Gluon, external asynchronous
compilation, custom repr/launch metadata, load/cache hooks and forced compiler
overrides use the original path. Actual autotuning trial closures enter an
exact-request ContextVar scope, including their thread task bodies.

## Optional preparation and current restriction

`snapshot.py`, `protocol.py`, `worker.py` and `service.py` implement explicit
source bundles, a new-interpreter compile worker, checked atomic manifests,
bounded task scheduling and an internal loader. Requests contain source,
retained constants, types and options, never invocation tensors or streams.
They do not import business modules or unpickle business functions.

Automatic background compilation/loading is **disabled**: `capabilities()` has
no qualified environment record. The validation covers compile-worker identity,
foreground multi-stream execution, and paired native loading correctness/latency
for eight cold objects. Shared initialization ownership across unhandled original
JIT entry points and the full shared-cache/failure-lifecycle matrix remain
unqualified. The per-object initialization lock does not establish per-LoadedKey
ownership across distinct CompiledKernel objects. No worker/thread starts on
a foreground miss or when an existing compatible version is selected under
this restriction. Foreground D/T reuse remains available.

Floating scalar data uses may enter D under the same structural rules as other
numeric scalars. This does not enable constexpr control/shape uses or arbitrary
unsupported operations.
This is a limited first delivery, not a claim that every M0–M4 acceptance row
or the background-switching promise has been qualified.

## Diagnostics and verification

`triton.backends.ascend.reuse.bridge.counters()` exposes in-memory counters;
`diagnostics()` returns JSON-compatible classification/plan evidence. Neither
returns business arguments. Native compile/cache events can be measured with
the existing Triton compilation listener.

Tests live in `third_party/ascend/unittest/reuse_ut`. Run host tests with
`pytest .../reuse_ut -k 'not npu'`. Run hardware acceptance in a fresh process
with both `TRITON_ASCEND_ENABLE_DYNAMIC_REUSE=1` and
`RUN_NPU_REUSE_TESTS=1`, selecting `test_npu.py` explicitly. Keep the background
capability gate closed until its additional acceptance has completed.

Public integration is delivered only through the Ascend patch. New package
discovery is maintained by `third_party/ascend/build/setup_patch.py` for both
the `reuse` and `reuse.dsl` packages. Development and release patch chains must
both be applied to clean public sources when validating a build.

See the [A5-37 validation report](../../unittest/reuse_ut/validation/A5-37_torch-213_20260930.md)
for scope, measurements and the remaining background-publication acceptance.
