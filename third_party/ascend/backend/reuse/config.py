import os
from functools import lru_cache

SCHEMA_VERSION = 1
RULE_VERSION = 10
ABI_VERSION = 2
ANALYSIS_CAPACITY = 128
ANALYSIS_NODE_BUDGET = 12000
LOOP_ITERATIONS = 16
MAX_READY = 64
MAX_FAMILY_READY = 8
MAX_TASKS = 32
FAILURE_THRESHOLD = 3


@lru_cache(maxsize=1)
def enabled():
    return os.environ.get("TRITON_ASCEND_ENABLE_DYNAMIC_REUSE", "0") == "1"


def supports_target(target):
    # D uses unannotated runtime arguments; the backend validates target/options.
    return target.backend == "npu"


def supports_tile_reuse(target, options):
    # The backend validates modes; pure SIMT has a separate execution contract.
    return supports_target(target) and options.compile_mode != "simt_only"
