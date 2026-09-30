"""Call guards for the implemented cross-tile launch recipe.

Dynamic scalar types, numeric operations and their errors belong to native JIT.
The index bounds below are specific to this T recipe, not D eligibility.
"""
TILE_INDEX_MIN, TILE_INDEX_MAX = -(2**31), 2**31 - 1


def _tile_index(value):
    return type(value) is int and TILE_INDEX_MIN <= value <= TILE_INDEX_MAX


def _offset_bounds(layout, grid):
    origin, *steps = layout
    spans = [(size - 1) * step for size, step in zip(grid, steps)]
    return origin + sum(min(0, span) for span in spans), origin + sum(max(0, span) for span in spans)


def _program_layout(expression, arguments, grid):
    """Evaluate a tile-independent affine base without enumerating the grid."""
    op, *operands = expression
    if op in ("lit", "arg", "num_programs"):
        value = operands[0] if op == "lit" else (arguments[operands[0]] if op == "arg" else grid[operands[0]])
        if not _tile_index(value):
            raise ValueError("unsupported base scalar")
        layout = (value, 0, 0, 0)
    elif op == "program_id":
        layout = (0, *(int(axis == operands[0]) for axis in range(3)))
    elif op in ("add", "sub", "mul"):
        left, right = (_program_layout(value, arguments, grid) for value in operands)
        if op == "add":
            layout = tuple(a + b for a, b in zip(left, right))
        elif op == "sub":
            layout = tuple(a - b for a, b in zip(left, right))
        elif not any(left[1:]):
            layout = tuple(left[0] * b for b in right)
        elif not any(right[1:]):
            layout = tuple(a * right[0] for a in left)
        else:
            raise ValueError("non-affine program base")
    else:
        raise ValueError("unmodeled program base")
    if any(not _tile_index(v) for v in _offset_bounds(layout, grid)):
        raise ValueError("program base can overflow")
    return layout


def _disjoint_programs(layout, grid, extent):
    span = extent
    for step, size in sorted((abs(step), size) for step, size in zip(layout[1:], grid) if size > 1):
        if step < span:
            return False
        span += step * (size - 1)
    return True


def adapt_tile(recipe, arguments, candidate_constants, grid, device, vector_limit, *, compile_mode):
    if not isinstance(grid, tuple) or not 1 <= len(grid) <= 3:
        return None
    if any(not _tile_index(g) or g <= 0 for g in grid):
        return None
    from .model import LocalTileRecipe
    if isinstance(recipe, LocalTileRecipe):
        if compile_mode == "simt_only":
            return None
        from .footprint import adapt_local_axis
        dimensions = (*grid, *((1, ) * (3 - len(grid))))
        return adapt_local_axis(recipe, arguments, candidate_constants, dimensions, device)
    if not recipe.local_loop and (any(g != 1 for g in grid[1:]) or vector_limit is None or grid[0] > vector_limit):
        return None
    grid = (*grid, *((1, ) * (3 - len(grid))))
    n, requested = (arguments[i] for i in (recipe.extent, recipe.tile))
    old = candidate_constants.get((recipe.tile, ))
    if any(not _tile_index(v) or v <= 0 for v in (n, requested, old)):
        return None
    from triton._utils import TRITON_MAX_TENSOR_NUMEL
    if any(b > TRITON_MAX_TENSOR_NUMEL for b in (requested, old)):
        return None
    # Match the Ascend frontend's arange restriction; SIMD permits other sizes.
    if compile_mode == "simt_only" and any(b & (b - 1) for b in (requested, old)):
        return None
    if n + max(old, requested) > TILE_INDEX_MAX:
        return None
    if recipe.bound is not None:
        blocks = arguments[recipe.bound]
        if not _tile_index(blocks) or blocks != (n + requested - 1) // requested:
            return None
    ranges = []
    offsets = dict(recipe.program_offsets)
    for index in recipe.pointers:
        tensor = arguments[index]
        try:
            layout = _program_layout(offsets.get(index, ("lit", 0)), arguments, grid)
        except (ValueError, TypeError, IndexError):
            return None
        low, high = _offset_bounds(layout, grid)
        if low < 0 or high + n + max(old, requested) > TILE_INDEX_MAX:
            return None
        if recipe.local_loop and index in recipe.stores and not _disjoint_programs(layout, grid, n):
            return None
        # Keep the supported host metadata protocol narrow and side-effect free.
        import torch
        if type(tensor) is not torch.Tensor or not tensor.is_contiguous() or tensor.numel() < high + n:
            return None
        if tensor.device.type != "npu" or tensor.device.index != device:
            return None
        start = tensor.data_ptr()
        width = tensor.element_size()
        mapping = (start + layout[0] * width, *(step * width for step in layout[1:]), width)
        ranges.append((index, start + low * width, start + (high + n) * width, mapping))
    for i, a, b, mapping in ranges:
        for j, c, d, other_mapping in ranges:
            if i == j or (i not in recipe.stores and j not in recipe.stores):
                continue
            # In-place access is safe only with the same element-to-byte mapping.
            if max(a, c) < min(b, d) and mapping != other_mapping:
                return None
    adapted = list(arguments)
    adapted[recipe.tile] = old
    if recipe.bound is not None:
        adapted[recipe.bound] = (n + old - 1) // old
    return tuple(adapted)
