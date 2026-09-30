"""Ascend policy behind the optional, backend-neutral JIT continuation."""
import os
from collections import OrderedDict
from threading import RLock

from . import config
from .binding import ArgumentBinder
from .context import InvocationFrame, requires_exact_request

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
    from .model import AutoReuseProfile
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

    def has_hooks(hook):
        return bool(hook.calls) if isinstance(hook, knobs.HookChain) else hook is not None

    if any((knobs.runtime.jit_cache_hook, knobs.runtime.jit_post_compile_hook,
            has_hooks(knobs.runtime.kernel_load_start_hook), has_hooks(knobs.runtime.kernel_load_end_hook),
            knobs.runtime.add_stages_inspection_hook, knobs.compilation.override, knobs.compilation.always_compile)):
        return True
    import torch
    capture = getattr(torch.npu, "is_current_stream_capturing", None)
    return capture is None or capture()


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
        self.runtime_environment = None
        self.capability = None

    def run(self, fn, args, kwargs, grid, warmup, device, stream, backend, jit_tail):

        def original(frame=None):
            return jit_tail(args, kwargs, grid, warmup, device, stream, frame=frame)

        if requires_original(fn, warmup) or os.getpid() != self.process:
            return original()
        # Match Python arguments before parsing options, as the original binder
        # does. Keep the option cache independent of dynamic semantic values.
        with self.lock:
            binder_key = ("binding", id(fn), id(fn.signature))
            binder = self.cache.get(binder_key)
            if binder is None:
                try:
                    binder = ArgumentBinder(fn.signature)
                except ValueError:
                    binder = None
                else:
                    self.cache[binder_key] = binder
        if binder is None:
            return original()
        bound, launch_options = binder(*args, **kwargs)
        arguments = tuple(bound.values())
        from ..compiler import NPUOptions
        from .identity import typed
        if any(name in NPUOptions.__dataclass_fields__ for name in fn.arg_names):
            return original()
        try:
            option_key = ("options", id(backend), typed(launch_options))
        except ValueError:
            return original()
        with self.lock:
            option_entry = self.cache.get(option_key)
            if option_entry is None:
                options = backend.parse_options(kwargs)
                option_entry = (options, repr(options))
                self.cache[option_key] = option_entry
                while len(self.cache) > config.ANALYSIS_CAPACITY:
                    self.cache.popitem(last=False)
            else:
                self.cache.move_to_end(option_key)
        options, option_token = option_entry
        if not config.supports_target(backend.target) or options.ir_override:
            return original()
        if self.device is not None and device != self.device:
            return original()
        from triton._C.libtriton import get_cache_invalidating_env_vars
        from triton.runtime.jit import compute_cache_key

        from .analysis import analyze, static_context, type_context
        from .dispatch import resolve_dispatch
        from .dsl.source import source_stamp
        from .identity import family_key, variant_key
        from .model import BuildSpec, PreparedLaunch
        from .registry import VariantRegistry
        from .transform import FrozenJITFunction, changed

        # No native specialization before independent binding and analysis.
        with self.lock:
            if self.registry is None:
                self.device = device
                self.environment = get_cache_invalidating_env_vars()
                self.registry = VariantRegistry()
                from .protocol import environment
                from .service import capabilities
                self.runtime_environment = environment(device, backend.target)
                self.capability = capabilities(self.runtime_environment)
        with self.lock:
            stamp_key = ("source-stamp", id(fn))
            stamp = self.cache.get(stamp_key)
            if stamp is None or not stamp.matches():
                stamp = source_stamp(fn)
                self.cache[stamp_key] = stamp
            versions = (config.SCHEMA_VERSION, config.RULE_VERSION, config.ABI_VERSION)
            key = (id(fn), stamp.token, versions, type_context(fn, arguments), static_context(fn, arguments),
                   repr(backend.target), option_token)
            entry = self.cache.get(key)
            if entry is None:
                self.registry.event("analysis_miss")
                profile = analyze(fn, bound, stamp.token,
                                  allow_tile_reuse=config.supports_tile_reuse(backend.target, options))
                owner = FrozenJITFunction.from_plan(profile.plan) if changed(profile.plan) else fn
                entry = (profile, owner)
                self.cache[key] = entry
                while len(self.cache) > config.ANALYSIS_CAPACITY:
                    self.cache.popitem(last=False)
            else:
                self.registry.event("analysis_hit")
                self.cache.move_to_end(key)
        profile, owner = entry
        if owner is fn and profile.recipe is None:
            self.registry.event("analysis_unknown")
            return original()
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
            return original()
        memory_key = compute_cache_key(key_cache, key_specialization, raw_options)
        spec_key = ("build", id(owner), profile.source_key, versions, device, memory_key, specialization_tag)
        with self.lock:
            spec = self.cache.get(spec_key)
            if spec is None:
                opts, signature, constants, attrs = owner._pack_args(actual_backend, kwargs, actual_bound,
                                                                     specialization, raw_options)
                source = owner.ASTSource(owner, signature, constants, attrs)
                options_hash_key = ("options-hash", option_token)
                options_hash = self.cache.get(options_hash_key)
                if options_hash is None:
                    options_hash = opts.hash()
                    self.cache[options_hash_key] = options_hash
                try:
                    family = family_key(profile, source, opts, target, self.environment, options_hash)
                except ValueError:
                    spec = None
                else:
                    spec = BuildSpec(owner, source, memory_key,
                                     variant_key(source, actual_backend, opts, self.environment, options_hash), family,
                                     opts, tuple(specialization))
                    self.cache[spec_key] = spec
                    while len(self.cache) > config.ANALYSIS_CAPACITY:
                        self.cache.popitem(last=False)
            else:
                self.cache.move_to_end(spec_key)
        if spec is None or self.registry.disabled(spec.family_key, spec.variant_key):
            return original()
        if type(grid) is list:
            grid = tuple(grid)
        load_context = (self.process, device)
        frame = InvocationFrame()
        decision = resolve_dispatch(self.registry, spec, profile, arguments, grid, load_context, actual_backend,
                                    self.runtime_environment["core_limit"][1],
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
                        self.preparation = PreparationService(self.registry, self.runtime_environment, self.capability)
                    self.preparation.try_submit(spec, load_context)
                else:
                    self.registry.event("queue_skipped")
            return kernel
        from triton.compiler.errors import CompilationError, MLIRCompilationError
        from triton.runtime.errors import OutOfResources
        try:
            return jit_tail(args, kwargs, grid, warmup, device, stream, owner=owner, bound=actual_bound,
                            specialization=specialization, raw_options=raw_options, memory_key=memory_key,
                            initialize=lambda kernel: self.registry.initialize(kernel, spec, load_context), frame=frame)
        except (CompilationError, MLIRCompilationError, OutOfResources):
            if owner is fn or frame.launch_started:
                raise
            self.registry.record_transform_failure(spec.family_key, spec.variant_key)
            self.registry.event("fallback")
            failed = owner.device_caches[device][0].pop(memory_key, None)
            if failed is not None:
                self.failed_objects.append(failed)
            # The original call is still the same invocation: no hooks or grid replay.
            # Its source/ABI has no D registration callback.
            return original(frame)
