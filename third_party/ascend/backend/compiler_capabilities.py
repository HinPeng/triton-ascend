# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Negotiate optional backend optimizations without modifying AscendNPU-IR.

Do not use capability probing to silently drop required semantics or ABI flags.
"""
import functools
from pathlib import Path
import re
import subprocess


@functools.lru_cache(maxsize=32)
def _supported_options(path, size, mtime_ns):
    # The file identity invalidates cached probes after an in-place upgrade.
    try:
        result = subprocess.run([path, "--help-hidden"], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"Cannot query NPU compiler capabilities: {path}") from error
    if result.returncode:
        raise RuntimeError(f"NPU compiler capability query failed: {path}: {result.stderr[-2000:]}")
    options = frozenset(re.findall(r"(?m)^\s*(--[a-zA-Z][a-zA-Z0-9-]*)(?=[=\s]|$)", result.stdout))
    if not options:
        raise RuntimeError(f"NPU compiler returned no option information: {path}")
    return options


def optimized_metaop_options(compiler_path, enabled):
    """Legacy compilers have no optimized-metaop path or corresponding switch."""
    path = Path(compiler_path).resolve()
    stamp = path.stat()
    if "--enable-optimized-metaop" not in _supported_options(str(path), stamp.st_size, stamp.st_mtime_ns):
        return []
    # New compilers must still receive False on older CANN versions.
    return [f"--enable-optimized-metaop={enabled}"]
