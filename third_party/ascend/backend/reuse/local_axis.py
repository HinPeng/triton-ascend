"""Use-site proof for masked axes tiled within programs or directly across a grid.

Shape/scheduling dependence is tracked separately from value dependence. The
proof does not recognize kernel names, parameter positions or whole functions.
Pointwise operations and associative accumulators compose through the same walk.
"""
import ast
import inspect
from dataclasses import dataclass, replace

from . import config
from .index_expr import AxisIdentity, LoopOrigin, grid_dependencies, has_index, normalize, schedule_dependent, terms
from .model import LocalTileRecipe, TileAccess
from .operations import resolve

ONE = ("lit", 1)
ZERO = ("lit", 0)
UNKNOWN = object()


class Unsupported(Exception):
    pass


class Returned(Exception):

    def __init__(self, value):
        self.value = value


@dataclass(frozen=True)
class Value:
    expression: object = None
    shape: tuple = ()
    schedule: bool = False
    padding: object = None
    pending: frozenset = frozenset()
    partial: bool = False
    raw: object = None
    known: object = UNKNOWN
    static_deps: frozenset = frozenset()
    items: object = None
    grid_uses: frozenset = frozenset()


def remap(expression, axes):
    if expression is None:
        return None
    if expression[0] in ("lane", "coordinate", "grid_coordinate"):
        return (*expression[:-1], axes[expression[-1]])
    return tuple(remap(v, axes) if isinstance(v, tuple) else v for v in expression)


class LocalAxis:

    def __init__(self, fn, tile, bound=None):
        self.fn, self.tile = fn, tile
        self.tile_expr = ("arg", tile)
        self.scope = fn.get_capture_scope()
        self.env = {name: Value(("arg", i), schedule=i == tile) for i, name in enumerate(fn.arg_names)}
        self.pointer_parameters = {
            i
            for i, name in enumerate(fn.arg_names)
            if hasattr((bound or {}).get(name), "data_ptr")
        }
        for i, name in enumerate(fn.arg_names):
            value = (bound or {}).get(name, UNKNOWN)
            if i != tile and fn.params[i].is_constexpr and type(value) in (bool, int, float):
                self.env[name] = replace(self.env[name], known=value, static_deps=frozenset((i, )))
        self.extent = None
        self.in_loop = False
        self.accesses, self.shapes = [], set()
        self.accumulators = {}
        self.loop_writes, self.defined = set(), set()
        self.remaining = config.ANALYSIS_NODE_BUDGET
        self.axis = None
        self.loops, self.index_checks = [], set()
        self.static_parameters = set()
        self.active_helpers = {id(fn)}
        self.frame_id = 0
        self.next_frame = 0
        self.helper_base_loop = False
        self.grid_axis = None
        self.observed_grid_uses = set()

    def op(self, node):
        import triton.language as tl
        obj = resolve(node, self.scope)
        if obj is range:
            return "range"
        for name in ("range", "cdiv", "arange", "program_id", "num_programs", "zeros", "full", "load", "store", "where",
                     "sum", "min", "max", "sqrt", "rsqrt", "exp", "log", "abs", "minimum", "maximum", "reshape"):
            if obj is not None and obj is getattr(tl, name, None):
                return name
        raise Unsupported("operation has no tile semantics")

    def align(self, values):
        rank = max((len(v.shape) for v in values), default=0)
        aligned = [
            replace(
                v, shape=(ONE, ) * (rank - len(v.shape)) + v.shape,
                expression=remap(self.resolve_grid_coordinate(v.expression),
                                 {i: i + rank - len(v.shape)
                                  for i in range(len(v.shape))}),
                raw=remap(v.raw, {i: i + rank - len(v.shape)
                                  for i in range(len(v.shape))}))
            for v in values
        ]
        shape = []
        for sizes in zip(*(v.shape for v in aligned)):
            nonunit = set(sizes) - {ONE}
            if len(nonunit) > 1:
                raise Unsupported("incompatible broadcast")
            shape.append(next(iter(nonunit), ONE))
        shape = tuple(shape)
        if shape.count(self.tile_expr) > 1:
            raise Unsupported("one traversal cannot cover two independent tile axes")
        if shape:
            self.shapes.add(shape)
        return aligned, shape

    def resolve_grid_coordinate(self, expression):
        if expression is None or self.grid_axis is None:
            return expression
        if expression[0] == "grid_coordinate":
            if expression[1].dimension != self.grid_axis:
                raise Unsupported("tile splits more than one grid axis")
            return ("coordinate", self.axis, expression[-1])
        return tuple(self.resolve_grid_coordinate(v) if isinstance(v, tuple) else v for v in expression)

    def infer_grid_extent(self, expression):
        if expression is None or expression[0] != "lt" or expression[1][0] != "grid_coordinate":
            return
        origin, extent = expression[1][1], expression[2]
        if schedule_dependent(extent, self.tile) or origin.dimension in grid_dependencies(extent):
            raise Unsupported("extent depends on grid partition")
        if self.loops or (self.extent is not None and self.extent != extent):
            raise Unsupported("conflicting tile traversal")
        if self.grid_axis is not None and self.grid_axis != origin.dimension:
            raise Unsupported("tile splits more than one grid axis")
        self.extent = extent
        self.axis = AxisIdentity(self.tile, extent)
        self.grid_axis = origin.dimension

    def combine(self, values, expression=None, padding=None):
        values, shape = self.align(values)
        return Value(expression, shape, any(v.schedule for v in values), padding,
                     frozenset().union(*(v.pending for v in values)),
                     grid_uses=frozenset().union(*(v.grid_uses for v in values)))

    def binary(self, op, left, right):
        if left.pending or right.pending:
            raise Unsupported("accumulator must be reduced before pointwise use")
        (left, right), shape = self.align((left, right))
        a, b = left.expression, right.expression
        expression = normalize(op, a, b, tile=self.tile)
        self.infer_grid_extent(expression)
        expression = self.resolve_grid_coordinate(expression)
        raw_a = left.raw if left.raw is not None else a
        raw_b = right.raw if right.raw is not None else b
        raw = (op, raw_a, raw_b) if raw_a is not None and raw_b is not None else None
        schedule = left.schedule or right.schedule
        if expression is not None:
            schedule = schedule_dependent(expression, self.tile)
        else:
            for operand in (raw_a, raw_b):
                if has_index(operand):
                    self.index_checks.add((operand, None))
        padding = None
        if left.padding is not None and right.padding is not None:
            if op == "add":
                padding = left.padding + right.padding
            elif op == "sub":
                padding = left.padding - right.padding
            elif op == "mul":
                padding = left.padding * right.padding
        known = UNKNOWN
        if type(left.known) in (bool, int, float) and type(right.known) in (bool, int, float):
            x, y = left.known, right.known
            operations = {
                "add": lambda: x + y, "sub": lambda: x - y, "mul": lambda: x * y, "floordiv": lambda: x // y, "lt":
                lambda: x < y, "gt": lambda: x > y, "le": lambda: x <= y, "ge": lambda: x >= y, "eq": lambda: x == y,
                "ne": lambda: x != y, "and": lambda: x & y
            }
            if op in operations:
                known = operations[op]()
        return Value(
            expression, shape, schedule, padding, left.pending | right.pending, raw=raw, known=known,
            static_deps=left.static_deps | right.static_deps,
            grid_uses=grid_dependencies(expression) if expression is not None else left.grid_uses | right.grid_uses)

    def shape(self, node):
        nodes = node.elts if isinstance(node, (ast.Tuple, ast.List)) else [node]
        values = tuple(self.expression(n) for n in nodes)
        self.observed_grid_uses.update(frozenset().union(*(v.grid_uses for v in values)))
        if any(v.shape or v.pending or v.expression is None for v in values):
            raise Unsupported("unmodeled shape")
        shape = tuple(v.expression for v in values)
        for value in values:
            if value.raw is not None:
                self.index_checks.add((value.raw, None))
        if shape.count(self.tile_expr) > 1:
            raise Unsupported("multiple local tile axes")
        self.shapes.add(shape)
        return shape

    def reshape(self, value, shape):
        old = [(i, n) for i, n in enumerate(value.shape) if n != ONE]
        new = [(i, n) for i, n in enumerate(shape) if n != ONE]
        if [n for _, n in old] != [n for _, n in new]:
            raise Unsupported("reshape mixes logical axes")
        axes = dict(zip([i for i, _ in old], [i for i, _ in new]))
        return replace(value, shape=shape, expression=remap(value.expression, axes), raw=remap(value.raw, axes))

    def expression(self, node):
        self.remaining -= 1
        if self.remaining < 0:
            raise Unsupported("analysis budget")
        if isinstance(node, ast.Constant):
            return Value(("lit", node.value), padding=node.value, known=node.value)
        if isinstance(node, ast.Name) and node.id in self.env:
            value = self.env[node.id]
            if value.partial or (self.in_loop and (value.pending or
                                                   (node.id in self.loop_writes and node.id not in self.defined))):
                raise Unsupported("loop state escapes its associative update")
            return value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return self.binary("sub", Value(ZERO, padding=0, known=0), self.expression(node.operand))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
            return self.expression(node.operand)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            value = self.expression(node.operand)
            if value.known is UNKNOWN or value.schedule:
                raise Unsupported("runtime condition")
            return Value(("lit", not value.known), known=not value.known, static_deps=value.static_deps)
        if isinstance(node, ast.BinOp):
            op = {
                ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div", ast.FloorDiv: "floordiv", ast.BitAnd:
                "and"
            }.get(type(node.op))
            if op is None:
                raise Unsupported("unmodeled arithmetic")
            return self.binary(op, self.expression(node.left), self.expression(node.right))
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            op = {ast.Lt: "lt", ast.Gt: "gt", ast.LtE: "le", ast.GtE: "ge", ast.Eq: "eq", ast.NotEq:
                  "ne"}.get(type(node.ops[0]))
            if op is not None:
                return self.binary(op, self.expression(node.left), self.expression(node.comparators[0]))
        if isinstance(node, ast.IfExp):
            return self.expression(node.body if self.condition(node.test) else node.orelse)
        if isinstance(node, ast.BoolOp):
            result = isinstance(node.op, ast.And)
            for child in node.values:
                result = self.condition(child)
                if result != isinstance(node.op, ast.And):
                    break
            return Value(("lit", result), known=result)
        if isinstance(node, (ast.Tuple, ast.List)):
            return Value(items=tuple(self.expression(n) for n in node.elts))
        if isinstance(node, ast.Subscript):
            value = self.expression(node.value)
            if value.items is not None and isinstance(node.slice, ast.Constant) and type(node.slice.value) is int:
                return value.items[node.slice.value]
            indices = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            shape, axes, old = [], {}, 0
            for index in indices:
                if isinstance(index, ast.Constant) and index.value is None:
                    shape.append(ONE)
                elif isinstance(index, ast.Slice) and index.lower is index.upper is index.step is None:
                    axes[old] = len(shape)
                    shape.append(value.shape[old])
                    old += 1
                else:
                    raise Unsupported("indexing changes the tile axis")
            while old < len(value.shape):
                axes[old] = len(shape)
                shape.append(value.shape[old])
                old += 1
            return replace(value, shape=tuple(shape), expression=remap(value.expression, axes),
                           raw=remap(value.raw, axes))
        if isinstance(node, ast.Call):
            return self.call(node)
        raise Unsupported("unmodeled expression")

    def masked(self, mask, coordinate):
        return mask is not None and ("lt", coordinate, coordinate[1].extent) in terms(mask, "and")

    def tile_dependent(self, expression):
        return expression == self.tile_expr or any(self.tile_dependent(v) for v in expression if isinstance(v, tuple))

    def condition(self, node):
        value = self.expression(node)
        if value.schedule or type(value.known) not in (bool, int, float) or value.pending:
            raise Unsupported("branch needs a tile-independent constexpr condition")
        self.static_parameters.update(value.static_deps)
        return bool(value.known)

    def assign(self, target, value):
        if isinstance(target, ast.Name):
            self.env[target.id] = value
            self.defined.add(target.id)
        elif isinstance(target,
                        (ast.Tuple, ast.List)) and value.items is not None and len(target.elts) == len(value.items):
            for child, item in zip(target.elts, value.items):
                self.assign(child, item)
        else:
            raise Unsupported("unsupported assignment target")

    def helper(self, fn, node):
        if id(fn) in self.active_helpers or len(self.active_helpers) >= 16:
            raise Unsupported("recursive or excessively deep JIT helper")
        arguments = [self.expression(n) for n in node.args]
        keywords = {k.arg: self.expression(k.value) for k in node.keywords}
        bound = fn.signature.bind(*arguments, **keywords)
        for name, parameter in fn.signature.parameters.items():
            if name not in bound.arguments:
                default = parameter.default
                if default is inspect.Parameter.empty or type(default) not in (bool, int, float):
                    raise Unsupported("unsupported helper default")
                bound.arguments[name] = Value(("lit", default), known=default, padding=default)
        saved = (self.env, self.scope, self.loop_writes, self.defined, self.frame_id, self.fn, self.in_loop,
                 self.helper_base_loop)
        self.next_frame += 1
        self.frame_id = self.next_frame
        self.env = dict(bound.arguments)
        for parameter in fn.params:
            if not parameter.is_constexpr:
                self.env[parameter.name] = replace(self.env[parameter.name], known=UNKNOWN)
        self.scope = fn.get_capture_scope()
        self.fn = fn
        self.helper_base_loop = self.in_loop
        self.loop_writes = set()
        self.defined = set(self.env)
        self.active_helpers.add(id(fn))
        try:
            self.statements(fn.parse().body[0].body)
        except Returned as result:
            return result.value
        finally:
            self.active_helpers.remove(id(fn))
            (self.env, self.scope, self.loop_writes, self.defined, self.frame_id, self.fn, self.in_loop,
             self.helper_base_loop) = saved
        return Value()

    def call(self, node):
        import triton.language as tl
        if any(k.arg is None for k in node.keywords):
            raise Unsupported("expanded call")
        kwargs = {k.arg: k.value for k in node.keywords}
        if resolve(node.func, self.scope) is float and len(node.args) == 1 and not kwargs and isinstance(
                node.args[0], ast.Constant) and node.args[0].value in ("inf", "-inf"):
            value = float(node.args[0].value)
            return Value(("lit", value), padding=value)
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("to", "reshape"):
            value = self.expression(node.func.value)
            if node.func.attr == "reshape" and len(node.args) == 1 and not kwargs:
                return self.reshape(value, self.shape(node.args[0]))
            if node.func.attr == "to" and len(node.args) == 1 and not kwargs:
                if not isinstance(resolve(node.args[0], self.scope), tl.dtype):
                    raise Unsupported("dynamic dtype")
                if value.pending:
                    raise Unsupported("cast before completed accumulation")
                # A cast's index precision/range is not part of the integer
                # normalization proof. Loaded data still propagates its shape.
                return replace(value, expression=None, raw=None, known=UNKNOWN)
            raise Unsupported("unmodeled tensor method")
        try:
            op = self.op(node.func)
        except Unsupported:
            from triton.runtime.jit import JITFunction
            fn = resolve(node.func, self.scope)
            if isinstance(fn, JITFunction):
                return self.helper(fn, node)
            raise
        if op == "cdiv" and len(node.args) == 2 and not kwargs:
            return self.binary("cdiv", self.expression(node.args[0]), self.expression(node.args[1]))
        if op in ("zeros", "full"):
            if set(kwargs) - {"dtype"} or len(node.args) != (1 if op == "zeros" else 2):
                raise Unsupported("unmodeled initializer")
            shape = self.shape(node.args[0])
            if "dtype" in kwargs and not isinstance(resolve(kwargs["dtype"], self.scope), tl.dtype):
                raise Unsupported("dynamic dtype")
            initial = 0 if op == "zeros" else self.expression(node.args[1]).padding
            if initial is None:
                raise Unsupported("nonconstant initializer")
            return Value(shape=shape, padding=initial)
        if op == "arange":
            if len(node.args) != 2 or kwargs or self.expression(node.args[0]).expression != ZERO:
                raise Unsupported("nonzero arange start")
            size = self.expression(node.args[1])
            if size.shape or size.expression is None:
                raise Unsupported("dynamic arange shape")
            self.shapes.add((size.expression, ))
            if size.raw is not None:
                self.index_checks.add((size.raw, None))
            return Value(("lane", size.expression, 0), (size.expression, ), size.schedule)
        if op in ("program_id", "num_programs"):
            if len(node.args) > 1 or set(kwargs) - {"axis"} or (node.args and kwargs):
                raise Unsupported("unmodeled program query")
            axis = self.expression(node.args[0] if node.args else kwargs["axis"]).expression
            if axis not in (("lit", 0), ("lit", 1), ("lit", 2)):
                raise Unsupported("dynamic program axis")
            return Value((op, axis[1]), grid_uses=frozenset((axis[1], )))
        if op in ("load", "store"):
            allowed = {"mask", "other"} if op == "load" else {"mask"}
            if len(node.args) != (1 if op == "load" else 2) or set(kwargs) - allowed:
                raise Unsupported("unmodeled memory effect")
            address = self.expression(node.args[0])
            mask = self.expression(kwargs["mask"])
            (address, mask), shape = self.align((address, mask))
            self.observed_grid_uses.update(address.grid_uses | mask.grid_uses)
            if address.expression is None or mask.expression is None or address.schedule or mask.schedule:
                raise Unsupported("tile affects addresses or masks")
            address_terms = terms(address.expression, "add")
            roots = [v for v in address_terms if v[0] == "arg"]
            typed_roots = [v for v in roots if v[1] in self.pointer_parameters]
            if len(typed_roots) == 1:
                roots = typed_roots
            if len(roots) != 1:
                raise Unsupported("unmodeled pointer")
            pointer = roots[0][1]
            address_terms.remove(roots[0])
            offset = ZERO
            for term in address_terms:
                offset = term if offset == ZERO else ("add", offset, term)
            coordinate = ("coordinate", self.axis, shape.index(self.tile_expr)) if self.tile_expr in shape else None
            if coordinate is not None and not self.masked(mask.expression, coordinate):
                raise Unsupported("local axis is not masked to its extent")
            self.accesses.append(TileAccess(pointer, offset, mask.expression, shape, op == "store"))
            if address.raw is not None:
                self.index_checks.add((address.raw, pointer))
            if mask.raw is not None:
                self.index_checks.add((mask.raw, None))
            if op == "store":
                value = self.expression(node.args[1])
                self.observed_grid_uses.update(value.grid_uses)
                _, output_shape = self.align((address, value))
                if value.schedule or value.pending or output_shape != shape:
                    raise Unsupported("tile state escapes to output")
                return Value()
            padding = self.expression(kwargs["other"]).padding if "other" in kwargs else None
            return Value(shape=shape, padding=padding, grid_uses=address.grid_uses | mask.grid_uses)
        if op in ("sum", "min", "max"):
            value = self.expression(node.args[0])
            axis_node = node.args[1] if len(node.args) > 1 else kwargs.get("axis")
            if axis_node is None or set(kwargs) - {"axis", "keep_dims"} or len(node.args) > 2:
                raise Unsupported("unmodeled reduction")
            axis = self.expression(axis_node).expression
            if axis is None or axis[0] != "lit" or type(axis[1]) is not int:
                raise Unsupported("dynamic reduction axis")
            if not -len(value.shape) <= axis[1] < len(value.shape):
                raise Unsupported("invalid reduction axis")
            axis = axis[1] % len(value.shape)
            pending = value.pending
            if value.shape[axis] == self.tile_expr:
                if not pending or any(self.accumulators[name] != op for name in pending):
                    raise Unsupported("partial reduction or incompatible accumulator")
                pending = frozenset()
            elif pending:
                raise Unsupported("reduction mixes accumulator lanes")
            keep = self.expression(kwargs["keep_dims"]).expression == ("lit", True) if "keep_dims" in kwargs else False
            shape = value.shape[:axis] + ((ONE, ) if keep else ()) + value.shape[axis + 1:]
            return Value(shape=shape, schedule=value.schedule, pending=pending, grid_uses=value.grid_uses)
        if op == "where":
            if len(node.args) != 3 or kwargs:
                raise Unsupported("unmodeled selection")
            values, shape = self.align([self.expression(n) for n in node.args])
            condition, left, right = values
            padding = left.padding if left.padding == right.padding else None
            if self.tile_expr in shape and self.masked(condition.expression,
                                                       ("coordinate", self.axis, shape.index(self.tile_expr))):
                padding = right.padding
            return self.combine(values, padding=padding)
        if op in ("sqrt", "rsqrt", "exp", "log", "abs", "minimum", "maximum"):
            if kwargs or len(node.args) != (2 if op in ("minimum", "maximum") else 1):
                raise Unsupported("unmodeled pointwise arguments")
            values = [self.expression(n) for n in node.args]
            if any(v.pending for v in values):
                raise Unsupported("nonlinear operation before completed reduction")
            padding = None
            if op in ("minimum", "maximum") and all(v.padding is not None for v in values):
                padding = (min if op == "minimum" else max)(v.padding for v in values)
            return self.combine(values, padding=padding)
        raise Unsupported("operation outside supported context")

    def update(self, name, contribution, op):
        state = self.env[name]
        value = self.expression(contribution)
        (state, value), shape = self.align((state, value))
        if not self.in_loop or self.tile_expr not in shape or state.shape != shape or value.schedule or value.pending:
            raise Unsupported("non-associative loop state")
        neutral = {"sum": 0, "max": -float("inf"), "min": float("inf")}[op]
        if (not state.pending and state.padding != neutral) or value.padding != neutral:
            raise Unsupported("accumulator padding is not neutral")
        key = (self.frame_id, name)
        if key in self.accumulators and self.accumulators[key] != op:
            raise Unsupported("mixed accumulator operations")
        self.accumulators[key] = op
        self.env[name] = Value(shape=shape, padding=neutral, pending=frozenset((key, )))

    def statements(self, body):
        for node in body:
            self.remaining -= 1
            if self.remaining < 0:
                raise Unsupported("analysis budget")
            if isinstance(node, ast.For):
                if self.in_loop or node.orelse or not isinstance(node.target, ast.Name):
                    raise Unsupported("nested or conditional tile loop")
                call = node.iter
                if not isinstance(call, ast.Call) or self.op(
                        call.func) != "range" or call.keywords or not 1 <= len(call.args) <= 3:
                    raise Unsupported("unmodeled loop")
                args = [self.expression(n) for n in call.args]
                if len(args) == 1:
                    start, extent, step = Value(ZERO), args[0], Value(ONE)
                elif len(args) == 2:
                    start, extent, step = *args, Value(ONE)
                else:
                    start, extent, step = args
                counts_tiles = False
                if extent.raw is not None:
                    self.index_checks.add((extent.raw, None))
                if step.raw is not None:
                    self.index_checks.add((step.raw, None))
                if step.expression == ONE and extent.expression is not None and extent.expression[0] == "cdiv" and (
                        extent.expression[2] == self.tile_expr):
                    extent = Value(extent.expression[1], schedule=self.tile_dependent(extent.expression[1]))
                    counts_tiles = True
                elif step.expression != self.tile_expr:
                    raise Unsupported("unmodeled tile-loop step")
                if start.expression != ZERO or extent.schedule or extent.shape:
                    raise Unsupported("loop is not a full local tile traversal")
                if extent.expression is None or (self.extent is not None and self.extent != extent.expression):
                    raise Unsupported("different local extents")
                self.extent = extent.expression
                self.axis = AxisIdentity(self.tile, self.extent)
                origin = LoopOrigin(self.axis, len(self.loops), counts_tiles, getattr(self.fn, "__name__", "kernel"),
                                    getattr(node, "lineno", 0))
                self.loops.append(origin)
                before = dict(self.env)
                self.env[node.target.id] = Value(("induction", origin), schedule=True)
                self.loop_writes = {
                    n.id
                    for statement in node.body
                    for n in ast.walk(statement)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                }
                self.defined = {node.target.id}
                self.in_loop = True
                self.statements(node.body)
                self.in_loop = False
                for name, value in list(self.env.items()):
                    if value.pending:
                        continue
                    if name not in before or value != before[name]:
                        self.env[name] = replace(value, partial=True)
            elif isinstance(node, ast.If):
                self.statements(node.body if self.condition(node.test) else node.orelse)
            elif isinstance(node, ast.Return):
                if self.frame_id == 0 or (self.in_loop and not self.helper_base_loop):
                    raise Unsupported("early return from kernel")
                raise Returned(self.expression(node.value) if node.value is not None else Value())
            elif isinstance(node, ast.Pass):
                continue
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                import triton.language as tl
                if resolve(node.annotation, self.scope) is not tl.constexpr:
                    raise Unsupported("unsupported local annotation")
                self.assign(node.target, self.expression(node.value))
            elif isinstance(node, ast.Assign) and (len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name)):
                value = self.expression(node.value)
                for target in node.targets:
                    self.assign(target, value)
            elif isinstance(node, ast.Assign):
                name = node.targets[0].id
                value = node.value
                if self.in_loop and isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add) and isinstance(
                        value.left, ast.Name) and value.left.id == name:
                    self.update(name, value.right, "sum")
                elif self.in_loop and isinstance(value, ast.Call) and len(
                        value.args) == 2 and not value.keywords and (isinstance(value.args[0], ast.Name)
                                                                     and value.args[0].id == name):
                    op = self.op(value.func)
                    if op not in ("minimum", "maximum"):
                        raise Unsupported("unmodeled accumulator")
                    self.update(name, value.args[1], "min" if op == "minimum" else "max")
                else:
                    self.env[name] = self.expression(value)
                self.defined.add(name)
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and isinstance(node.op, ast.Add):
                self.update(node.target.id, node.value, "sum")
                self.defined.add(node.target.id)
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                self.expression(node.value)
            else:
                raise Unsupported("unmodeled statement")

    def run(self):
        self.statements(self.fn.parse().body[0].body)
        if self.extent is None or not any(a.write for a in self.accesses):
            return None
        if self.grid_axis is not None:
            if self.loops or self.grid_axis in self.observed_grid_uses:
                raise Unsupported("changed program partition affects computation")
            if any(access.write and self.tile_expr not in access.shape for access in self.accesses):
                raise Unsupported("per-program output is not an elementwise tiled output")
        # The runtime proof checks signed 32-bit intermediates. Explicit narrow
        # or unsigned index annotations need their own arithmetic model.
        def argument_indices(expression):
            if expression[0] == "arg":
                return {expression[1]}
            return set().union(*(argument_indices(v) for v in expression if isinstance(v, tuple)))

        index_parameters = set()
        for expression, pointer in self.index_checks:
            index_parameters.update(argument_indices(expression) - {pointer})
        for index in index_parameters:
            parameter = self.fn.params[index]
            annotation = getattr(parameter, "annotation_type", "")
            if not parameter.is_constexpr and annotation and annotation not in ("i32", "i64"):
                raise Unsupported("index arithmetic width is not modeled")
        return LocalTileRecipe(self.tile, self.extent, tuple(self.accesses), tuple(sorted(self.shapes)),
                               axes=(self.axis, ), loops=tuple(self.loops),
                               index_checks=tuple(sorted(self.index_checks, key=repr)),
                               static_parameters=tuple(sorted(self.static_parameters)), grid_axis=self.grid_axis,
                               local_loop=self.grid_axis is None,
                               rule="MaskedGridTileV1" if self.grid_axis is not None else "MaskedLocalAxisV1")


def match_local_axis(fn, bound=None):
    for position, parameter in enumerate(fn.params):
        if not parameter.is_constexpr:
            continue
        try:
            recipe = LocalAxis(fn, position, bound).run()
            if recipe is not None:
                return recipe
        except (Unsupported, KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
            continue
    return None
