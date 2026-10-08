"""Versioned compile-only requests and independently validated artifact manifests."""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

from . import config
from .identity import digest
from .snapshot import bundle, decode, encode, rebuild


def environment(device, target):
    import torch_npu
    import triton
    from triton import knobs
    from triton.runtime import driver

    utils = driver.active.utils
    return {
        "device": device,
        "target": [target.backend, target.arch, target.warp_size],
        "visible": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "python": list(sys.version_info[:2]),
        "triton": triton.__version__,
        "torch_npu": torch_npu.__version__,
        "cann": os.environ.get("ASCEND_HOME_PATH"),
        "cache": str(Path(knobs.cache.dir).resolve()),
        "physical_cores": list(utils.get_device_core()),
        "core_limit": [utils.get_aicore_num(), utils.get_aivector_core_num()],
    }


def build_request(spec, env, generation, attempt=0):
    return {
        "schema": config.SCHEMA_VERSION, "variant_key": spec.variant_key, "generation": generation, "attempt": attempt,
        "source_kind": "reuse" if hasattr(spec.source.fn, "reuse_identity") else "original", "source":
        bundle(spec.source.fn), "signature": encode(spec.source.signature), "constants": encode(spec.source.constants),
        "attrs": encode(spec.source.attrs), "options": encode(spec.options.__dict__), "environment": env
    }


def rebuild_source(request):
    from triton.compiler import ASTSource

    from .transform import ReuseASTSource

    if request["schema"] != config.SCHEMA_VERSION:
        raise ValueError("protocol version differs")
    fn = rebuild(request["source"])
    if request["source_kind"] not in ("original", "reuse"):
        raise ValueError("unknown source kind")
    cls = ASTSource if request["source_kind"] == "original" else ReuseASTSource
    return cls(fn, decode(request["signature"]), decode(request["constants"]), decode(request["attrs"]))


def file_record(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("artifact outside fixed cache root")
    data = path.read_bytes()
    return {"path": str(path), "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def publish_manifest(request, kernel):
    root = request["environment"]["cache"]
    files = {name: file_record(path, root) for name, path in kernel.metadata_group.items()}
    if not any(n.endswith(".npubin") for n in files) or not any(n.endswith(".json") for n in files):
        raise ValueError("incomplete compiler artifact group")
    manifest = {
        "schema": config.SCHEMA_VERSION, "variant_key": request["variant_key"], "source_digest":
        digest(request["source"]), "environment": request["environment"], "files": files, "generation":
        request["generation"], "attempt": request["attempt"]
    }
    directory = Path(root) / "reuse-manifests" / request["variant_key"]
    directory.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=directory, prefix="manifest-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(manifest, stream, sort_keys=True)
        os.replace(temporary, directory / "manifest.json")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return str(directory / "manifest.json")


def read_manifest(path, request):
    manifest = json.loads(Path(path).read_text())
    for field in ("schema", "variant_key", "generation", "attempt", "environment"):
        if manifest[field] != request[field]:
            raise ValueError("manifest mismatch: " + field)
    if manifest["source_digest"] != digest(request["source"]):
        raise ValueError("source bundle differs")
    files = manifest["files"]
    if not any(n.endswith(".npubin") for n in files) or not any(n.endswith(".json") for n in files):
        raise ValueError("incomplete manifest")
    for record in files.values():
        if file_record(record["path"], request["environment"]["cache"]) != record:
            raise ValueError("artifact digest differs")
    metadata = json.loads(Path(next(r["path"] for n, r in files.items() if n.endswith(".json"))).read_text())
    if metadata["hash"] != request["variant_key"]:
        raise ValueError("metadata identity differs")
    return {name: record["path"] for name, record in files.items()}
