"""Cheap live dependency stamps; AST traversal happens only on source changes."""
import ast
import builtins
import types
from dataclasses import dataclass

from ..identity import digest
from ..operations import operation, resolve


def path(node):
    if isinstance(node, ast.Name):
        return (node.id, )
    if isinstance(node, ast.Attribute):
        base = path(node.value)
        return (*base, node.attr) if base else ()
    return ()


def lookup(fn, route):
    name = route[0]
    nonlocal_names = fn.fn.__code__.co_freevars
    if name in nonlocal_names and fn.fn.__closure__:
        value = fn.fn.__closure__[nonlocal_names.index(name)].cell_contents
    else:
        value = fn.__globals__.get(name, vars(builtins).get(name))
    for name in route[1:]:
        value = vars(value).get(name) if isinstance(value, types.ModuleType) else None
    return value


@dataclass(frozen=True)
class SourceStamp:
    records: tuple
    token: str

    def matches(self):
        return all(fn.src == source and all(lookup(fn, route) is value
                                            for route, value in references)
                   for fn, source, references in self.records)


def source_stamp(fn):
    from triton.runtime.jit import JITFunction

    records, visited = [], set()

    def visit(current):
        if id(current) in visited:
            return
        visited.add(id(current))
        scope = current.get_capture_scope()
        references = []
        for node in ast.walk(current.parse()):
            if not isinstance(node, ast.Call):
                continue
            obj = resolve(node.func, scope)
            route = path(node.func)
            if route and obj is not None:
                references.append((route, obj))
            if isinstance(obj, JITFunction) and operation(obj) is None:
                visit(obj)
        records.append((current, current.src, tuple(references)))

    visit(fn)
    return SourceStamp(tuple(records), digest(tuple((f.cache_key, src) for f, src, _ in records)))
