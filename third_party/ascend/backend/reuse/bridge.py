"""Ascend policy behind the optional, backend-neutral JIT continuation."""
import os
from collections import OrderedDict
from threading import RLock

from . import config
from .binding import ArgumentBinder
from .context import InvocationFrame, requires_exact_request
from .debug import log_bypass, log_profile
from .model import AutoReuseProfile

_handler = None
_handler_lock = RLock()


def get_handler():
    global _handler
    if _handler is None:
        with _handler_lock:
            if _handler is None:
                _handler = ReuseHandler()
    return _handler


def counters():
    return {} if _handler is None or _handler.registry is None else _handler.registry.snapshot()


def diagnostics():
    """Host-only, JSON-compatible evidence; never return invocation arguments."""
    if _handler is None:
        return {"counts": {}, "profiles": []}
    with _handler.lock:
        profiles = {}
        for entry in _handler.cache.values():
            if not isinstance(entry, tuple) or not entry or not isinstance(entry[0], AutoReuseProfile):
                continue
            p = entry[0]
            profiles[p.source_key] = {
                "source":
                p.source_key,
                "parameters": [{"position": d.position, "classification": d.classification.value, "reasons": d.reasons}
                               for d in p.decisions],
                "dynamic_slots":
                p.plan.dynamic,
                "local_constexpr_lines":
                p.plan.local_lines,
                "tile_rule":
                None if p.recipe is None else p.recipe.rule,
                "abi_version":
                config.ABI_VERSION,
            }
        return {
            "counts": counters(), "profiles": list(profiles.values()), "background":
            None if _handler.capability is None else _handler.capability.reason
        }


def requires_original(fn, warmup):
    from triton import knobs
    from triton.runtime import _async_compile

    if warmup or requires_exact_request() or fn.is_gluon() or fn._repr or fn.launch_metadata:
        return True
    if _async_compile.active_mode.get() is not None:
        return True

    # Load hooks observe actual synchronous initialization, including transformed
    # binaries. READY hits do not load again and must not replay those hooks.
    # Compilation callbacks can alter compilation or interrupt it after side
    # effects; their exception boundaries are not covered by this continuation.
    if any((knobs.runtime.jit_cache_hook, knobs.runtime.jit_post_compile_hook,
            knobs.runtime.add_stages_inspection_hook, knobs.compilation.listener is not None,
            knobs.compilation.override, knobs.compilation.always_compile)):
        return True
    return False


class ReuseHandler:

    def __init__(self):
        self.registry = None
        self.cache = OrderedDict()
        self.lock = RLock()
        self.device = None
        self.process = os.getpid()
        self.environment = None
        self.failed_objects = []
        self.preparation = None
        self.capability = None

    def _cached(self, key):
        """Return an entry and mark it recently used. The caller holds self.lock."""
        value = self.cache.get(key)
        if value is not None:
            self.cache.move_to_end(key)
        return value

    def _remember(self, key, value):
        """Store an entry without capacity eviction. The caller holds self.lock."""
        self.cache[key] = value
        return value

    def _analysis_entry(self, fn, bound, source_key, target, option_token, *, allow_tile_reuse):
        """Reuse value-independent D plans and their JIT owners; caller holds self.lock."""
        from .analysis import analyze, static_context, type_context
        from .transform import FrozenJITFunction, changed

        arguments = tuple(bound.values())
        versions = (config.SCHEMA_VERSION, config.RULE_VERSION, config.ABI_VERSION)
        types = type_context(fn, arguments)
        context = (id(fn), source_key, versions, types, repr(target), option_token, allow_tile_reuse)
        values = static_context(fn, arguments)
        masks_key = ("analysis-masks", context)
        masks = self._cached(masks_key) or ()
        for dynamic in masks:
            key = ("analysis", context, dynamic, tuple(v for v in values if v[0] not in dynamic))
            entry = self._cached(key)
            if entry is not None:
                self.registry.event("analysis_hit")
                if dynamic:
                    self.registry.event("analysis_fast_hit")
                return entry

        self.registry.event("analysis_miss")
        profile = analyze(fn, bound, source_key, allow_tile_reuse=allow_tile_reuse)
        log_profile(fn, profile)
        owner = FrozenJITFunction.from_plan(profile.plan) if changed(profile.plan) else fn
        # Uses interprets both branches symbolically, without reading scalar values.
        # A completed D plan is therefore valid for every value of its dynamic slots
        # in this type context. Keep all other constexprs exact. Tile recipes can
        # depend on concrete values during matching and retain the full key.
        dynamic = frozenset(profile.plan.dynamic) if profile.recipe is None else frozenset()
        key = ("analysis", context, dynamic, tuple(v for v in values if v[0] not in dynamic))
        entry = self._remember(key, (profile, owner))
        if dynamic not in masks:
            self._remember(masks_key, (*masks, dynamic))
        return entry

    def run(self, fn, args, kwargs, grid, warmup, device, stream, backend, jit_tail):

        def original(frame=None, *, reason):
            log_bypass(fn, reason)
            return jit_tail(args, kwargs, grid, warmup, device, stream, frame=frame)

        if requires_original(fn, warmup):
            return original(reason="EXACT_REQUEST_REQUIRED: warmup, autotuning, hooks or custom JIT behavior")
        if os.getpid() != self.process:
            return original(reason="PROCESS_CHANGED")
        # Match Python arguments before parsing options, as the original binder
        # does. Keep the option cache independent of dynamic semantic values.
        with self.lock:
            binder_key = ("binding", id(fn), id(fn.signature))
            binder = self._cached(binder_key)
            if binder is None:
                try:
                    binder = self._remember(binder_key, ArgumentBinder(fn.signature))
                except ValueError:
                    binder = None
        if binder is None:
            return original(reason="ARGUMENT_BINDER_UNSUPPORTED")
        bound, launch_options = binder(*args, **kwargs)
        arguments = tuple(bound.values())

        from ..compiler import NPUOptions
        from .identity import typed

        if any(name in NPUOptions.__dataclass_fields__ for name in fn.arg_names):
            return original(reason="PARAMETER_NAME_CONFLICTS_WITH_COMPILER_OPTION")
        try:
            option_key = ("options", id(backend), typed(launch_options))
        except ValueError:
            return original(reason="LAUNCH_OPTIONS_NOT_SERIALIZABLE")
        with self.lock:
            option_entry = self._cached(option_key)
            if option_entry is None:
                options = backend.parse_options(kwargs)
                option_entry = self._remember(option_key, (options, repr(options)))
        options, option_token = option_entry
        if not config.supports_target(backend.target):
            return original(reason="TARGET_UNSUPPORTED")
        if options.ir_override:
            return original(reason="IR_OVERRIDE_ENABLED")
        if self.device is not None and device != self.device:
            return original(reason="DEVICE_CHANGED")

        from triton._C.libtriton import get_cache_invalidating_env_vars
        from triton.runtime.jit import compute_cache_key

        from .dispatch import resolve_dispatch
        from .dsl.source import source_stamp
        from .identity import family_key, variant_key
        from .model import BuildSpec, PreparedLaunch
        from .registry import VariantRegistry

        # No native specialization before independent binding and analysis.
        with self.lock:
            if self.registry is None:
                from .service import capabilities

                self.device = device
                self.environment = get_cache_invalidating_env_vars()
                self.registry = VariantRegistry()
                # The device environment is probed only when preparation is allowed.
                self.capability = capabilities()
        with self.lock:
            stamp_key = ("source-stamp", id(fn))
            stamp = self._cached(stamp_key)
            if stamp is None or not stamp.matches():
                stamp = self._remember(stamp_key, source_stamp(fn))
            versions = (config.SCHEMA_VERSION, config.RULE_VERSION, config.ABI_VERSION)
            entry = self._analysis_entry(fn, bound, stamp.token, backend.target, option_token,
                                         allow_tile_reuse=config.supports_tile_reuse(backend.target, options))
        profile, owner = entry
        if owner is fn and profile.recipe is None:
            self.registry.event("analysis_unknown")
            return original(reason="NO_ELIGIBLE_CONSTEXPR_PARAMETERS")
        _, key_cache, target, actual_backend, specialize = owner.device_caches[device]
        actual_bound, specialization, raw_options = specialize(*args, **kwargs)
        try:
            specialization_tag = typed(specialization)
            # Retain the native key algorithm, with typed values in the private D
            # namespace. Python equality otherwise aliases True/1 and +/-0.0 in
            # the native specialization memo. T-only keeps its original key.
            key_specialization = ([(ty, typed(value))
                                   for ty, value in specialization] if owner is not fn else specialization)
        except ValueError:
            return original(reason="SPECIALIZATION_NOT_SERIALIZABLE")
        memory_key = compute_cache_key(key_cache, key_specialization, raw_options)
        spec_key = ("build", id(owner), profile.source_key, versions, device, memory_key, specialization_tag)
        with self.lock:
            spec = self._cached(spec_key)
            if spec is None:
                opts, signature, constants, attrs = owner._pack_args(actual_backend, kwargs, actual_bound,
                                                                     specialization, raw_options)
                source = owner.ASTSource(owner, signature, constants, attrs)
                options_hash_key = ("options-hash", option_token)
                options_hash = self._cached(options_hash_key)
                if options_hash is None:
                    options_hash = self._remember(options_hash_key, opts.hash())
                try:
                    family = family_key(profile, source, opts, target, self.environment, options_hash)
                except ValueError:
                    spec = None
                else:
                    spec = self._remember(
                        spec_key,
                        BuildSpec(owner, source, memory_key,
                                  variant_key(source, actual_backend, opts, self.environment, options_hash), family,
                                  opts, tuple(specialization)))
        if spec is None:
            return original(reason="REUSE_FAMILY_NOT_SERIALIZABLE")
        if self.registry.disabled(spec.family_key, spec.variant_key):
            return original(reason="TRANSFORM_FAILURE_THRESHOLD_REACHED")
        if type(grid) is list:
            grid = tuple(grid)
        load_context = (self.process, device)
        frame = InvocationFrame()
        decision = resolve_dispatch(self.registry, spec, profile, arguments, grid, load_context, actual_backend,
                                    resolve_grid=lambda: frame.resolve_grid(grid, actual_bound))

        if isinstance(decision, PreparedLaunch):
            if decision.grid is not None:
                grid = decision.grid
                frame.select_grid(grid)
            kernel = jit_tail(args, kwargs, grid, warmup, device, stream, owner=owner,
                              prepared=decision.executable.kernel, bound=dict(zip(fn.arg_names,
                                                                                  decision.arguments)), frame=frame)
            if decision.executable.variant_key != spec.variant_key:
                from .service import PreparationService

                if self.capability.compile:
                    if self.preparation is None:
                        from .protocol import environment

                        self.preparation = PreparationService(self.registry, environment(device, backend.target),
                                                              self.capability)
                    self.preparation.try_submit(spec, load_context)
                else:
                    self.registry.event("queue_skipped")
            return kernel

        from triton.compiler.errors import CompilationError, MLIRCompilationError
        from triton.runtime.errors import OutOfResources

        def initialize(kernel):
            try:
                return self.registry.initialize(kernel, spec, load_context)
            finally:
                # Once an external load callback has run, a fallback would
                # replay its effects, even when the error looks like a compiler
                # or resource failure. Keep the original exception unchanged.
                frame.load_hook_started |= getattr(kernel, "_load_hook_started", False)

        try:
            return jit_tail(args, kwargs, grid, warmup, device, stream, owner=owner, bound=actual_bound,
                            specialization=specialization, raw_options=raw_options, memory_key=memory_key,
                            initialize=initialize, frame=frame)
        except (CompilationError, MLIRCompilationError, OutOfResources) as error:
            if owner is fn or frame.launch_started or frame.load_hook_started:
                raise
            self.registry.record_transform_failure(spec.family_key, spec.variant_key)
            self.registry.event("fallback")
            failed = owner.device_caches[device][0].pop(memory_key, None)
            if failed is not None:
                self.failed_objects.append(failed)
            # The original call is still the same invocation: no hooks or grid replay.
            # Its source/ABI has no D registration callback.
            return original(frame, reason=f"TRANSFORM_COMPILE_FAILED: {type(error).__name__}")
