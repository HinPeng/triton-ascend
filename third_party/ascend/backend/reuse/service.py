"""Bounded optional preparation. Capabilities require separate hardware evidence."""
import atexit
import json
import os
import selectors
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock

from . import config
from .protocol import build_request, read_manifest, rebuild_source
from .snapshot import SourceNotSerializable


@dataclass(frozen=True)
class Capabilities:
    compile: bool = False
    initialize: bool = False
    reason: str = "SHARED_JIT_INITIALIZATION_NOT_QUALIFIED"


def capabilities(environment=None):
    # No inferred capability from SoC name alone: CANN/native loading and cache
    # races must pass the independent G03 matrix before adding an allow record.
    return Capabilities()


class CompileWorker:

    def __init__(self):
        read_fd, write_fd = os.pipe()
        self.process = subprocess.Popen([sys.executable, "-m", "triton.backends.ascend.reuse.worker",
                                         str(write_fd)], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL, text=True, pass_fds=(write_fd, ))
        os.close(write_fd)
        self.replies = os.fdopen(read_fd)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.replies, selectors.EVENT_READ)

    def compile(self, request):
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        if not self.selector.select(timeout=300):
            self.close()
            raise TimeoutError("compile worker timed out")
        line = self.replies.readline()
        if not line:
            raise BrokenPipeError("compile worker exited")
        response = json.loads(line)
        if "error" in response:
            raise ValueError(response["error"] + ": " + response["message"])
        return response["manifest"]

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        self.selector.close()
        self.replies.close()
        self.process.stdin.close()


class DeviceLoader:

    def __init__(self, registry):
        self.registry = registry
        self.retained = []

    def initialize(self, request, manifest, spec, context):
        import torch
        from triton.compiler.compiler import CompiledKernel
        group = read_manifest(manifest, request)
        source = rebuild_source(request)
        # Optional background loading is unavailable unless *all* entry points
        # sharing a LoadedKey participate in registry initialization ownership.
        with torch.npu.device(context[1]):
            kernel = CompiledKernel(source, group, request["variant_key"])
            try:
                kernel = self.registry.initialize(kernel, spec, context)
            except Exception:
                self.retained.append(kernel)  # do not release possibly live native handles
                raise
            return self.registry.register_ready_if_eligible(kernel, spec, context)


class PreparationService:

    def __init__(self, registry, environment, capability=None):
        self.registry = registry
        self.environment = environment
        self.capability = capability or capabilities(environment)
        self.lock = Lock()
        self.tasks = {}
        self.errors = {}
        self.generation = 0
        self.closed = False
        self.worker = None
        self.loader = DeviceLoader(registry)
        self.compiler_executor = None
        self.loader_executor = None
        atexit.register(self.close)

    def try_submit(self, spec, context):
        if not self.capability.compile:
            self.registry.event("queue_skipped")
            return False
        with self.lock:
            if (self.closed or spec.variant_key in self.tasks or len(self.tasks) >= config.MAX_TASKS
                    or self.registry.lookup_exact_ready(spec.variant_key, context) is not None):
                self.registry.event("queue_skipped")
                return False
            try:
                request = build_request(spec, self.environment, self.generation)
            except SourceNotSerializable:
                self.registry.event("source_not_serializable")
                return False
            # The queue stores descriptors, never invocation frames or tensors.
            self.tasks[spec.variant_key] = "QUEUED"
            if self.compiler_executor is None:
                self.compiler_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ascend-reuse-compile")
            self.compiler_executor.submit(self._prepare, request, spec, context)
            return True

    def _prepare(self, request, spec, context):
        key = request["variant_key"]
        try:
            with self.lock:
                if self.closed or request["generation"] != self.generation:
                    self.registry.event("stale_result")
                    return
                self.tasks[key] = "BUILDING"
            for attempt in range(2):
                request["attempt"] = attempt
                try:
                    if self.worker is None:
                        self.worker = CompileWorker()
                    manifest = self.worker.compile(request)
                    read_manifest(manifest, request)
                    break
                except (OSError, TimeoutError):
                    if self.worker is not None:
                        self.worker.close()
                        self.worker = None
                    if attempt:
                        raise
                    self.registry.event("retry")
            with self.lock:
                if self.closed or request["generation"] != self.generation:
                    self.registry.event("stale_result")
                    return
                self.tasks[key] = "ARTIFACT_READY"
                if self.capability.initialize:
                    if self.loader_executor is None:
                        self.loader_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ascend-reuse-load")
                    self.loader_executor.submit(self._initialize, request, manifest, spec, context)
        except (ValueError, OSError, TimeoutError) as error:
            with self.lock:
                self.tasks[key] = "BUILD_FAILED"
                self.errors[key] = (type(error).__name__, str(error))
            self.registry.event("background_build_failed")

    def _initialize(self, request, manifest, spec, context):
        with self.lock:
            if self.closed or request["generation"] != self.generation:
                self.registry.event("stale_result")
                return
            self.tasks[request["variant_key"]] = "INITIALIZING"
        try:
            ready = self.loader.initialize(request, manifest, spec, context)
            state = "READY" if ready is not None else "INIT_FAILED"
        except Exception as error:  # noqa: BLE001 - background failure is recorded, never replayed
            state = "INIT_FAILED"
            with self.lock:
                self.errors[request["variant_key"]] = (type(error).__name__, str(error))
            self.registry.event("background_init_failed")
        with self.lock:
            self.tasks[request["variant_key"]] = state

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.generation += 1
        if self.worker is not None:
            self.worker.close()
        for executor in (self.compiler_executor, self.loader_executor):
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
