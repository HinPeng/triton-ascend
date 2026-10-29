# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Byte-domain proof for static tensor-axis-aware access summaries."""
from dataclasses import dataclass

from .block_batching_guard import GuardFailure, I32_MAX, checked, integer


@dataclass(frozen=True)
class AffineND:
    offset: int
    coefficients: tuple

    @property
    def uniform(self):
        return not any(self.coefficients)

    def add(self, other, sign=1):
        return AffineND(self.offset + sign * other.offset,
                        tuple(a + sign * b for a, b in zip(self.coefficients, other.coefficients)))

    def scale(self, value):
        return AffineND(self.offset * value, tuple(c * value for c in self.coefficients))


class ByteDomain:

    def __init__(self, arguments, grid, access):
        self.arguments, self.grid, self.access = arguments, grid, access
        self.shape = tuple(integer(n) for n in access.get("shape", ()))
        if len(self.shape) > 8 or any(n <= 0 for n in self.shape):
            raise GuardFailure("UNSUPPORTED_ACCESS_RANK")
        self.checks = []
        self.budget = 1024
        self.loop_axes = {}
        loops = access.get("loops", ())
        if not isinstance(loops, (tuple, list)) or len(loops) > 8:
            raise GuardFailure("INVALID_LOOP_DOMAIN")
        counts = []
        for loop in loops:
            identity = integer(loop["id"])
            if identity < 0 or identity in self.loop_axes:
                raise GuardFailure("INVALID_LOOP_DOMAIN")
            domain = ByteDomain(arguments, grid, {"shape": []})
            bounds = []
            for field in ("lower", "upper", "step"):
                value = domain.expression(loop[field])
                if not value.uniform:
                    raise GuardFailure("NON_UNIFORM_LOOP_DOMAIN")
                bounds.append(value.offset)
            for value, bits in domain.checks:
                checked(*domain.bounds(value, domain.full), bits)
            lower, upper, step = bounds
            bits = self.bits(loop["bits"])
            checked(min(lower, upper), max(lower, upper), bits)
            if step <= 0:
                raise GuardFailure("UNPROVEN_POSITIVE_LOOP_STEP")
            count = max(0, (upper - lower + step - 1) // step)
            # Keep the original induction increment representable as well.
            checked(lower, lower + count * step, bits)
            position = 3 + len(self.shape) + len(counts)
            self.loop_axes[identity] = (position, lower, step, bits)
            counts.append(count)
        self.loop_counts = tuple(counts)
        self.full = (*grid, *self.shape, *counts)

    def constant(self, value):
        return AffineND(value, (0, ) * len(self.full))

    @staticmethod
    def bounds(value, counts):
        lo = hi = value.offset
        for coefficient, count in zip(value.coefficients, counts):
            delta = coefficient * (count - 1)
            lo += min(0, delta)
            hi += max(0, delta)
        return lo, hi

    @staticmethod
    def bits(value):
        if type(value) is not int or value not in (1, 32, 64):
            raise GuardFailure("INVALID_INDEX_WIDTH")
        return value

    def expression(self, node):
        self.budget -= 1
        if self.budget < 0:
            raise GuardFailure("ANALYSIS_BUDGET_EXCEEDED")
        if not isinstance(node, list) or not node:
            raise GuardFailure("INVALID_GUARD_EXPRESSION")
        name = node[0]
        if name in ("arg", "const"):
            value = integer(self.arguments[node[1]]) if name == "arg" else integer(node[1])
            checked(value, value, self.bits(node[2]))
            return self.constant(value)
        if name == "grid":
            axis = integer(node[1])
            if not 0 <= axis < 3:
                raise GuardFailure("INVALID_PROGRAM_AXIS")
            return self.constant(self.grid[axis])
        if name == "iv":
            identity = integer(node[1])
            if identity not in self.loop_axes:
                raise GuardFailure("UNSUPPORTED_LOOP_CAPTURE")
            axis, lower, step, bits = self.loop_axes[identity]
            if self.bits(node[2]) != bits:
                raise GuardFailure("INVALID_LOOP_DOMAIN")
            coefficients = [0] * len(self.full)
            coefficients[axis] = step
            return AffineND(lower, tuple(coefficients))
        if name in ("pid", "coord"):
            axis = integer(node[1]) + (3 if name == "coord" else 0)
            if name == "pid" and not 0 <= axis < 3:
                raise GuardFailure("INVALID_PROGRAM_AXIS")
            if name == "coord" and (not 3 <= axis < 3 + len(self.shape) or self.full[axis] != integer(node[2])):
                raise GuardFailure("INVALID_LOCAL_AXIS")
            coefficients = [0] * len(self.full)
            coefficients[axis] = 1
            return AffineND(0, tuple(coefficients))
        if name in ("minsi", "maxsi"):
            lhs, rhs = self.expression(node[2]), self.expression(node[3])
            if not lhs.uniform or not rhs.uniform:
                raise GuardFailure("NON_UNIFORM_SCALAR_BOUND")
            result = self.constant((min if name == "minsi" else max)(lhs.offset, rhs.offset))
            self.checks.append((result, self.bits(node[1])))
            return result
        if name in ("addi", "subi", "muli"):
            lhs, rhs = self.expression(node[2]), self.expression(node[3])
            if name == "muli":
                if not lhs.uniform and not rhs.uniform:
                    raise GuardFailure("NON_AFFINE_ADDRESS")
                result = rhs.scale(lhs.offset) if lhs.uniform else lhs.scale(rhs.offset)
            else:
                result = lhs.add(rhs, -1 if name == "subi" else 1)
            self.checks.append((result, self.bits(node[1])))
            return result
        if name in ("extsi", "extui", "trunci"):
            result = self.expression(node[2])
            if name == "extui":
                source_bits = self.bits(node[3])
                lo, hi = self.bounds(result, self.full)
                if lo < 0 <= hi:
                    raise GuardFailure("NON_AFFINE_UNSIGNED_EXTENSION")
                if hi < 0:
                    result = result.add(self.constant(1 << source_bits))
            self.checks.append((result, self.bits(node[1])))
            return result
        raise GuardFailure("UNSUPPORTED_GUARD_EXPRESSION")

    def predicates(self, mask):
        if mask is None or mask == ["unknown"]:
            return []
        if mask == ["true"]:
            return []
        if mask == ["false"]:
            return [("false", None, None, 0)]
        if mask[0] == "and":
            return self.predicates(mask[1]) + self.predicates(mask[2])
        if mask[0] in ("slt", "ult", "eq"):
            return [(mask[0], self.expression(mask[1]), self.expression(mask[2]), self.bits(mask[3]))]
        # No use of an unmodeled predicate to shrink the original domain.
        return []

    def interval(self, pointer):
        if any(count == 0 for count in self.loop_counts):
            return None
        width = self.access["bytes"]
        volatile = self.access.get("volatile", False)
        if type(volatile) is not bool:
            raise GuardFailure("INVALID_MEMORY_ARGUMENT")
        exclusive = self.access["write"] or volatile
        overlap_reason = "OVERLAPPING_VOLATILE_READS" if volatile else "OVERLAPPING_BLOCK_WRITES"
        if type(width) is not int or width not in (1, 2, 4, 8):
            raise GuardFailure("INVALID_MEMORY_ARGUMENT")
        address = self.expression(self.access["offset"])
        predicates = self.predicates(self.access.get("mask"))
        steps = [self.expression(node) for node in self.access.get("pointer_checks", ())]
        # Validate all source arithmetic before a mask is allowed to trim it.
        for expression, bits in self.checks:
            lo, hi = self.bounds(expression, self.full)
            checked(lo, hi, bits)
        for step in steps:
            lo, hi = self.bounds(step, self.full)
            checked(pointer + lo, pointer + hi, 64, unsigned=True)
        local_strides = address.coefficients[3:3 + len(self.shape)]
        if any(local_strides) and width not in local_strides:
            raise GuardFailure("NON_UNIT_COLUMN_STRIDE")
        counts = list(self.full)
        merged = {}
        for predicate, index, bound, bits in predicates:
            if predicate == "false":
                return None
            if not bound.uniform:
                continue
            if predicate == "ult":
                lo, hi = self.bounds(index, self.full)
                if lo < 0:
                    continue
                limit = bound.offset % (1 << bits)
            else:
                limit = bound.offset
            if index.uniform:
                truth = index.offset == limit if predicate == "eq" else index.offset < limit
                if not truth:
                    return None
                continue
            nonzero = [axis for axis, c in enumerate(index.coefficients) if c]
            if index.offset < 0 or any(c < 0 for c in index.coefficients):
                continue
            remaining = limit - index.offset
            if predicate == "eq":
                # A nonnegative affine sum equals zero only at zero on each
                # participating axis. Other equalities retain the full box.
                if remaining == 0:
                    for axis in nonzero:
                        counts[axis] = min(counts[axis], 1)
                continue
            if remaining <= 0:
                return None
            if len(nonzero) == 1 and index.coefficients[nonzero[0]] == 1:
                axis = nonzero[0]
                counts[axis] = min(counts[axis], remaining)
            elif len(nonzero) == 2:
                grids = [axis for axis in nonzero if axis < 3]
                locals_ = [axis for axis in nonzero if 3 <= axis < 3 + len(self.shape)]
                if len(grids) != 1 or len(locals_) != 1:
                    continue
                g, local = grids[0], locals_[0]
                block = self.full[local]
                if (index.coefficients[g] != block or index.coefficients[local] != 1
                        or address.coefficients[g] != address.coefficients[local] * block):
                    continue
                total = min(remaining, counts[g] * block)
                key = (g, local)
                merged[key] = min(merged.get(key, total), total)
                counts[g] = min(counts[g], (total + block - 1) // block)
                counts[local] = min(counts[local], total)
        covered = set()
        dimensions = []
        for (g, local), count in merged.items():
            if g in covered or local in covered:
                continue
            covered.update((g, local))
            dimensions.append((address.coefficients[local], count))
        dimensions += [(stride, counts[axis])
                       for axis, stride in enumerate(address.coefficients)
                       if axis not in covered]
        if any(count <= 0 for _, count in dimensions):
            return None
        for stride, count in dimensions:
            if stride < 0:
                raise GuardFailure("NEGATIVE_STRIDE")
            if exclusive and count > 1 and stride == 0:
                raise GuardFailure(overlap_reason)
        span = width - 1
        for stride, count in sorted((s, n) for s, n in dimensions if s and n > 1):
            if exclusive and stride <= span:
                raise GuardFailure(overlap_reason)
            span += stride * (count - 1)
        start, end = pointer + address.offset, pointer + address.offset + span + 1
        checked(start, end, 64, unsigned=True)
        if start >= end:
            raise GuardFailure("INVALID_MEMORY_INTERVAL")
        return start, end, self.access["write"]


def check_guard_v2(guard, arguments, grid, factor):
    try:
        if type(guard.get("version")) is not int or guard["version"] != 2 or not guard.get("accesses"):
            raise GuardFailure("INVALID_GUARD_SCHEMA")
        grid = tuple(integer(n) for n in grid)
        if len(grid) != 3 or any(n < 0 or n > I32_MAX for n in grid):
            raise GuardFailure("GRID_OUT_OF_RANGE")
        if any(n == 0 for n in grid):
            return False, "EMPTY_GRID"
        axis = guard["axis"]
        if type(axis) is not int or axis not in (0, 1, 2) or type(factor) is not int or factor < 2:
            raise GuardFailure("INVALID_BATCH_REQUEST")
        tasks = 1
        for d, n in enumerate(grid):
            tasks *= (n + factor - 1) // factor if d == axis else n
        if tasks > I32_MAX:
            raise GuardFailure("G0_TASK_COUNT_OVERFLOW")
        ranges = []
        for access in guard["accesses"]:
            value = arguments[access["root"]]
            pointer = integer(value.data_ptr()) if hasattr(value, "data_ptr") else integer(value)
            if not 0 < pointer < (1 << 64):
                raise GuardFailure("INVALID_MEMORY_ARGUMENT")
            result = ByteDomain(arguments, grid, access).interval(pointer)
            if result is not None:
                ranges.append(result)
        for i, (lo, hi, write) in enumerate(ranges):
            for a, b, other_write in ranges[:i]:
                if (write or other_write) and lo < b and a < hi:
                    raise GuardFailure("WRITE_ALIAS")
        return True, "PROVEN_AT_RUNTIME"
    except (GuardFailure, KeyError, IndexError, TypeError, OverflowError) as error:
        return False, str(error)
