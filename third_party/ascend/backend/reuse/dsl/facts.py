from dataclasses import dataclass


def kind_of(value):
    if hasattr(value, "dtype") and hasattr(value, "data_ptr"):
        return "ptr:" + str(value.dtype).removeprefix("torch.")
    if isinstance(value, float):
        return "float"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    return "unknown"


def constant_value(value):
    import triton.language as tl
    if isinstance(value, tl.constexpr):
        value = value.value
    return Value(kind=kind_of(value))


@dataclass(frozen=True)
class FactResult:
    facts: tuple
    complete: bool
    diagnostics: tuple
    source_spans: tuple = ()
    dependency_ids: tuple = ()


@dataclass(frozen=True)
class Value:
    deps: frozenset = frozenset()
    runtime: bool = False
    kind: str = "int"


def merge(*values, runtime=False, kind=None):
    return Value(frozenset().union(*(v.deps for v in values)), runtime or any(v.runtime for v in values), kind
                 or (values[0].kind if values else "int"))
