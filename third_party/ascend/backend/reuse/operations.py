"""Operation identities and the first, deliberately finite semantic vocabulary."""
import builtins
import types


def resolve(node, scope):
    import ast
    if isinstance(node, ast.Name):
        return scope.get(node.id, vars(builtins).get(node.id))
    if isinstance(node, ast.Attribute):
        base = resolve(node.value, scope)
        # Never execute arbitrary user descriptors while analyzing a kernel.
        if isinstance(base, types.ModuleType):
            return vars(base).get(node.attr)
    return None


def operation(obj):
    import triton.language as tl
    for name in ("arange", "program_id", "num_programs", "load", "store", "where", "cast", "full", "zeros", "reshape",
                 "broadcast_to", "expand_dims", "static_range", "range", "static_assert", "sum", "min", "max", "gather",
                 "cdiv", "minimum", "maximum", "pointer_type"):
        if obj is getattr(tl, name, None) and obj is not None:
            return name
    if obj is range:
        return "range"
    return None


STATIC_ARGUMENTS = {
    "arange": ((0, 1), ("start", "end")),
    "program_id": ((0, ), ("axis", )),
    "num_programs": ((0, ), ("axis", )),
    "full": ((0, 2), ("shape", "dtype")),
    "zeros": ((0, 1), ("shape", "dtype")),
    "reshape": ((1, 2), ("shape", "can_reorder")),
    "broadcast_to": ((1, ), ("shape", )),
    "expand_dims": ((1, ), ("axis", )),
    "cast": ((1, 2, 3), ("dtype", "bitcast", "fp_downcast_rounding")),
    "sum": ((1, 2), ("axis", "keep_dims")),
    "min": ((1, 2, 3, 4), ("axis", "keep_dims", "return_indices", "return_indices_tie_break_left")),
    "max": ((1, 2, 3, 4), ("axis", "keep_dims", "return_indices", "return_indices_tie_break_left")),
    "minimum": ((2, ), ("propagate_nan", )),
    "maximum": ((2, ), ("propagate_nan", )),
    "gather": ((2, ), ("axis", )),
    "load": ((3, 4, 5, 6, 7), ("cache_modifier", "eviction_policy", "volatile", "boundary_check", "padding_option")),
    "store": ((3, 4, 5), ("cache_modifier", "eviction_policy", "boundary_check")),
}
