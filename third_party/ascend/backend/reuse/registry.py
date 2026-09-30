from collections import Counter
from threading import RLock

from . import config
from .model import ExecutableVariant


class VariantRegistry:

    def __init__(self):
        self.lock = RLock()
        self.exact = {}
        self.families = {}
        self.counts = Counter()
        self.failures = Counter()
        self.initialization_locks = {}

    def initialize(self, kernel, spec, load_context):
        key = (spec.variant_key, load_context)
        with self.lock:
            lock = self.initialization_locks.setdefault(key, RLock())
        with lock:
            ready = self.lookup_exact_ready(spec.variant_key, load_context)
            if ready is not None:
                self.register_ready_if_eligible(ready.kernel, spec, load_context)
                return ready.kernel
            was_initialized = getattr(kernel, "_initialization_complete", False)
            kernel._init_handles()
            if not was_initialized:
                self.event("native_init")
            self.register_ready_if_eligible(kernel, spec, load_context)
            return kernel

    def event(self, name):
        with self.lock:
            self.counts[name] += 1

    def lookup_exact_ready(self, variant_key, load_context):
        with self.lock:
            return self.exact.get((variant_key, load_context))

    def lookup_family_ready(self, family_key, load_context):
        with self.lock:
            return tuple(v for v in self.families.get(family_key, ()) if v.load_context == load_context)

    def has_registration(self, executable, family_key):
        with self.lock:
            return any(v is executable for v in self.families.get(family_key, ()))

    def register_ready_if_eligible(self, kernel, spec, load_context):
        if (not getattr(kernel, "_initialization_complete", False) or not kernel.module or not kernel.function
                or not callable(kernel._run) or kernel.hash != spec.variant_key
                or kernel.src.hash() != spec.source.hash() or kernel.src.signature != spec.source.signature
                or kernel.src.constants != spec.source.constants or kernel.src.attrs != spec.source.attrs):
            self.event("registration_skipped")
            return None
        with self.lock:
            failure_key = (spec.family_key, spec.variant_key)
            if self.failures[failure_key] >= config.FAILURE_THRESHOLD:
                self.counts["registration_skipped"] += 1
                return None
            key = (spec.variant_key, load_context)
            existing = self.exact.get(key)
            family = self.families.get(spec.family_key, ())
            if existing is not None:
                self.counts["registration_duplicate"] += 1
                if existing not in family and len(family) < config.MAX_FAMILY_READY:
                    self.families[spec.family_key] = (*family, existing)
                return existing
            if len(self.exact) >= config.MAX_READY or len(family) >= config.MAX_FAMILY_READY:
                self.counts["registration_skipped"] += 1
                return None
            value = ExecutableVariant(kernel, kernel.src, spec.variant_key, spec.family_key, load_context,
                                      len(self.exact))
            self.exact[key] = value
            self.families[spec.family_key] = (*family, value)
            self.counts["ready_registered"] += 1
            self.failures.pop(failure_key, None)
            return value

    def record_transform_failure(self, family, variant_key):
        with self.lock:
            key = (family, variant_key)
            self.failures[key] += 1
            self.counts["transform_failure"] += 1
            if self.failures[key] >= config.FAILURE_THRESHOLD:
                self.counts["circuit_open"] += 1

    def disabled(self, family, variant_key):
        with self.lock:
            return self.failures[(family, variant_key)] >= config.FAILURE_THRESHOLD

    def snapshot(self):
        with self.lock:
            return dict(self.counts)
