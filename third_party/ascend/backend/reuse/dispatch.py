from .footprint import adapt_grid
from .guards import adapt_tile
from .model import JitTailRequest, PreparedLaunch


def resolve_dispatch(registry, spec, profile, arguments, grid, load_context, backend, *, resolve_grid=None):
    exact = registry.lookup_exact_ready(spec.variant_key, load_context)

    def adapt(candidate):
        # Registration proof and compilation compatibility are separate checks.
        if not registry.has_registration(candidate, spec.family_key):
            return None
        if candidate.variant_key == spec.variant_key:
            return PreparedLaunch(candidate, arguments)
        if profile.recipe is None:
            return None
        candidate_grid = grid
        if callable(candidate_grid) and resolve_grid is not None:
            candidate_grid = resolve_grid()
        if type(candidate_grid) is list:
            candidate_grid = tuple(candidate_grid)
        actual = adapt_tile(profile.recipe, arguments, candidate.source.constants, candidate_grid, load_context[1],
                            compile_mode=spec.options.compile_mode)
        if actual is None:
            return None
        selected_grid = None
        if getattr(profile.recipe, "grid_axis", None) is not None:
            selected_grid = adapt_grid(profile.recipe, arguments, candidate.source.constants, candidate_grid)
            if selected_grid is None:
                return None
            # An explicit compiler grid contract must remain exact.
            original_dimensions = (*candidate_grid, *((1, ) * (3 - len(candidate_grid))))
            selected_dimensions = (*selected_grid, *((1, ) * (3 - len(selected_grid))))
            if selected_dimensions != original_dimensions and getattr(spec.options, "grid_num_tiles", None) is not None:
                return None
        _, _, _, _, binder = spec.owner.device_caches[load_context[1]]
        # Only specialization policy is needed here. Serialized compiler options
        # contain derived/deprecated fields that are not user launch keywords.
        options = {"compile_mode": spec.options.compile_mode}
        _, specialization, _ = binder(*actual, **options)
        # Options were checked for the desired BuildSpec. Recheck the adapted
        # ABI without repeating option normalization on a compatible READY hit.
        from triton._utils import find_paths_if, get_iterable_path
        types = [item[0] for item in specialization]
        signature = dict(zip(spec.owner.arg_names, types))
        constants = {
            path: get_iterable_path(actual, path)
            for path in find_paths_if(types, lambda _, value: value == "constexpr")
        }
        attributes = [item[1] for item in specialization]
        attrs = {
            path: backend.parse_attr(get_iterable_path(attributes, path))
            for path in find_paths_if(attributes, lambda _, value: isinstance(value, str))
        }
        if (signature != candidate.source.signature or constants != candidate.source.constants
                or attrs != candidate.source.attrs):
            return None
        return PreparedLaunch(candidate, actual, selected_grid)

    if exact is not None:
        result = adapt(exact)
        if result is not None:
            registry.event("ready_exact_hit")
            return result
        registry.event("candidate_rejected")
    registry.event("exact_unavailable")
    for candidate in registry.lookup_family_ready(spec.family_key, load_context):
        if candidate is exact:
            continue
        result = adapt(candidate)
        if result is not None:
            registry.event("ready_compatible_hit")
            return result
        registry.event("candidate_rejected")
    registry.event("family_exhausted")
    registry.event("jit_tail_miss")
    return JitTailRequest(spec)
