# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Evaluate one unified plan. Device SSA indices are never read on the host."""
from .block_batching_guard import GuardFailure, I32_MAX, integer
from .block_batching_guard_v2 import ByteDomain, check_layout_guards


def check_guard_v3(guard, arguments, grid, factor):
    try:
        if (type(guard.get('version')) is not int or guard['version'] != 3
                or guard.get('plan_kind') != 'unified'
                or guard.get('semantic_contract_version') != 4
                or guard.get('memory_policy') != 'disjoint_or_terminal_store'
                or 'ordered_effects' in guard or 'noninterference_pairs' in guard
                or type(guard.get('factor')) is not int or guard['factor'] != factor):
            raise GuardFailure('INVALID_GUARD_SCHEMA')
        grid = tuple(integer(n) for n in grid)
        if len(grid) != 3 or any(n < 0 or n > I32_MAX for n in grid):
            raise GuardFailure('GRID_OUT_OF_RANGE')
        if any(n == 0 for n in grid):
            return False, 'EMPTY_GRID'
        axis = guard['axis']
        if type(axis) is not int or axis not in (0, 1, 2) or type(factor) is not int or not 2 <= factor <= 64:
            raise GuardFailure('INVALID_BATCH_REQUEST')
        tasks = 1
        for d, n in enumerate(grid):
            tasks *= (n + factor - 1) // factor if d == axis else n
        if tasks > I32_MAX:
            raise GuardFailure('G0_TASK_COUNT_OVERFLOW')
        accesses, obligations = guard['accesses'], guard['obligations']
        if not isinstance(accesses, list) or not accesses or not isinstance(obligations, list) or len(accesses) != len(obligations):
            raise GuardFailure('INVALID_PROOF_SCHEMA')
        has_writes = any(access['write'] for access in accesses)
        write_ids = [i for i, access in enumerate(accesses) if access['write']]
        device_ids = [i for i, access in enumerate(accesses) if access['device_address']]
        terminal = guard['terminal_read_ids']
        if (not isinstance(terminal, list)
                or any(type(i) is not int for i in terminal)):
            raise GuardFailure('INVALID_EFFECT_DEPENDENCY')
        ranges = []
        for identity, (access, proof) in enumerate(zip(accesses, obligations)):
            if (not isinstance(proof, dict) or set(proof) != {'access_id', 'rule', 'proof'}
                    or type(proof['access_id']) is not int or proof['access_id'] != identity
                    or type(access.get('device_address')) is not bool):
                raise GuardFailure('INVALID_PROOF_SCHEMA')
            fields = {'root', 'shape', 'offset', 'mask', 'bytes', 'write',
                      'pointer_checks', 'loops', 'volatile', 'device_address'}
            if (not isinstance(access, dict) or set(access) != fields
                    or type(access['root']) is not int or access['root'] < 0
                    or type(access['write']) is not bool or type(access['volatile']) is not bool
                    or type(access['bytes']) is not int or access['bytes'] not in (1, 2, 4, 8)
                    or not isinstance(access['shape'], list) or len(access['shape']) > 8
                    or any(type(n) is not int or n <= 0 for n in access['shape'])
                    or not isinstance(access['pointer_checks'], list)
                    or not isinstance(access['loops'], list)):
                raise GuardFailure('INVALID_EFFECT_SCHEMA')
            value = arguments[access['root']]
            pointer = integer(value.data_ptr()) if hasattr(value, 'data_ptr') else integer(value)
            if not 0 < pointer < (1 << 64):
                raise GuardFailure('INVALID_MEMORY_ARGUMENT')
            if access['device_address']:
                if (proof['rule'] != 'ReplicateEffect' or proof['proof'] != 'lane_correspondence'
                        or access['write'] is not False or access['volatile'] is not False
                        or access['offset'] is not None or access['pointer_checks']):
                    raise GuardFailure('INVALID_EFFECT_PROOF')
                continue
            if proof['rule'] == 'ReplicateEffect':
                if access['write'] or access['volatile'] or proof['proof'] not in ('byte_domain', 'lane_correspondence'):
                    raise GuardFailure('INVALID_EFFECT_PROOF')
                if proof['proof'] == 'lane_correspondence' and not has_writes:
                    continue
            elif (proof['proof'] != 'byte_domain'
                    or proof['rule'] not in ('SupportedEffect', 'ShareInvariantLoad', 'PreserveVolatile')
                    or (proof['rule'] == 'ShareInvariantLoad' and (access['write'] or access['volatile']))
                    or (proof['rule'] == 'PreserveVolatile' and not access['volatile'])):
                raise GuardFailure('INVALID_EFFECT_PROOF')
            if access['offset'] is None:
                raise GuardFailure('INVALID_EFFECT_PROOF')
            interval = ByteDomain(arguments, grid, access).interval(pointer)
            if interval is not None:
                ranges.append(interval)
        if terminal:
            if (len(write_ids) != 1 or terminal != device_ids
                    or any(i >= write_ids[0] for i in terminal)
                    or any(access['volatile'] for access in accesses)):
                raise GuardFailure('INVALID_EFFECT_DEPENDENCY')
        elif has_writes and device_ids:
            raise GuardFailure('UNPROVEN_MEMORY_NONINTERFERENCE')
        for layout in guard.get('layout_checks', []):
            if accesses[layout['access_id']]['device_address']:
                raise GuardFailure('INVALID_LAYOUT_PROOF')
        check_layout_guards(guard, arguments, grid, factor)
        for i, (lo, hi, write) in enumerate(ranges):
            for a, b, other_write in ranges[:i]:
                if (write or other_write) and lo < b and a < hi:
                    raise GuardFailure('WRITE_ALIAS')
        return True, 'PROVEN_AT_RUNTIME'
    except (GuardFailure, KeyError, IndexError, TypeError, OverflowError) as error:
        return False, str(error)


def plan_digest(guard):
    import hashlib
    import json
    payload = {key: value for key, value in guard.items() if key != 'plan_fingerprint'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def bind_plan(guard, source):
    import hashlib
    guard['source_fingerprint'] = hashlib.sha256(source.encode()).hexdigest()
    guard['implementation_version'] = 20
    guard['plan_fingerprint'] = plan_digest(guard)


def validate_plan_binding(guard):
    if not isinstance(guard, dict):
        return False
    try:
        return (type(guard.get('version')) is int and guard['version'] == 3
                and type(guard.get('implementation_version')) is int and guard['implementation_version'] == 20
                and guard.get('semantic_contract_version') == 4
                and guard.get('memory_policy') == 'disjoint_or_terminal_store'
                and isinstance(guard.get('terminal_read_ids'), list)
                and 'ordered_effects' not in guard and 'noninterference_pairs' not in guard
                and isinstance(guard.get('source_fingerprint'), str)
                and len(guard['source_fingerprint']) == 64
                and guard.get('plan_fingerprint') == plan_digest(guard))
    except (TypeError, ValueError):
        return False
