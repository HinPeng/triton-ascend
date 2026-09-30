"""A fixed inner reduction may carry row-local state, never state between row tasks."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import triton.language as tl
from test_tile_rules import tensors as tensor_fixture
from triton.backends.ascend.reuse.analysis import analyze
from triton.backends.ascend.reuse.guards import adapt_tile
from triton.backends.ascend.reuse.local_axis import LocalAxis, match_tile_axes
from triton.backends.ascend.reuse.model import ParameterClass

tensors = tensor_fixture


def kernel(change=None):
    source = ast.unparse(ast.parse(Path(__file__).with_name('_rmsnorm_kernel.py').read_text()))
    if change:
        assert change[0] in source
        source = source.replace(*change)
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    names = tuple(a.arg for a in node.args.args)
    return SimpleNamespace(
        arg_names=names, params=tuple(
            SimpleNamespace(name=n, num=i, is_constexpr=i in (8, 9), annotation_type='', do_not_specialize=False)
            for i, n in enumerate(names)), parse=lambda: ast.Module(body=[node], type_ignores=[]),
        get_capture_scope=lambda: {'tl': tl})


def arguments(tensors, rows=17, columns=385, m=4, n=128, inplace=False):
    pitch = columns + 16
    x = tensors(65536, width=4, count=rows * pitch)
    y = x if inplace else tensors(1048576, width=4, count=rows * pitch)
    w = tensors(2097152, width=4, count=columns)
    for value in (x, y, w):
        value.dtype = 'float32'
    return (x, y, w, pitch, pitch, rows, columns, 1e-5, m, n)


def proof(args, change=None):
    fn = kernel(change)
    return match_tile_axes(fn, dict(zip(fn.arg_names, args)))


@pytest.mark.parametrize('old,new', [(2, 4), (4, 2), (4, 8), (8, 4)])
@pytest.mark.parametrize('grid', [(1, ), (3, ), (32, ), (3, 1, 1)])
@pytest.mark.parametrize('shape', [(1, 31), (17, 384), (17, 385)])
@pytest.mark.parametrize('inplace', [False, True])
def test_row_tile_reuse_keeps_fixed_column_reduction(tensors, old, new, grid, shape, inplace):
    args = arguments(tensors, *shape, m=new, inplace=inplace)
    recipe = proof(args)
    assert recipe is not None and recipe.tiles == (8, )
    assert 9 in recipe.static_parameters
    assert adapt_tile(recipe, args, {(8, ): old, (9, ): 128}, grid, 0, compile_mode='simd') == (*args[:8], old, 128)
    assert adapt_tile(recipe, args, {(8, ): old, (9, ): 64}, grid, 0, compile_mode='simd') is None


def test_classification_admits_m_only(tensors):
    args = arguments(tensors)
    fn = kernel()
    p = analyze(fn, dict(zip(fn.arg_names, args)), 'rmsnorm')
    assert p.recipe.tiles == (8, )
    assert p.decisions[0].classification is ParameterClass.ScheduleReusable
    assert p.decisions[1].classification is not ParameterClass.ScheduleReusable
    assert 9 not in p.plan.dynamic


@pytest.mark.parametrize('change', [
    ('ss_acc += tl.sum(x * x, axis=1)', 'ss_acc += tl.sum(x * x, axis=0)'),
    ('ss_acc += tl.sum(x * x, axis=1)', 'ss_acc += tl.sum(x * x, axis=1) + BLOCK_SIZE_M'),
    ('ss_acc += tl.sum(x * x, axis=1)', 'ss_acc += tl.sum(x * x, axis=1)\n            ss_acc = ss_acc * 2'),
    ('ss_acc += tl.sum(x * x, axis=1)',
     'ss_acc += tl.sum(x * x, axis=1)\n            ss_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)'),
    ('ss_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)',
     'ss_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32) + row_task_id'),
    ('ss_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)',
     'ss_acc = ss_acc + tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)'),
    ('range(0, n_cols, BLOCK_SIZE_N)', 'range(0, n_cols + BLOCK_SIZE_M, BLOCK_SIZE_N)'),
    ('rrms[:, None]', 'tl.sum(rrms, axis=0)'),
])
def test_unproved_recurrences_and_row_mixing_decline(tensors, change):
    args = arguments(tensors)
    assert proof(args, change) is None


def test_sum_initialized_outside_row_tasks_is_not_a_row_local_sum(tensors):
    fn = kernel()
    tree = fn.parse()
    body = tree.body[0].body
    loop = next(n for n in body if isinstance(n, ast.For))
    initial = next(n for n in loop.body if isinstance(n, ast.Assign) and n.targets[0].id == 'ss_acc')
    loop.body.remove(initial)
    body.insert(body.index(loop), initial)
    args = arguments(tensors)
    from triton.backends.ascend.reuse.local_axis import Unsupported
    with pytest.raises(Unsupported):
        LocalAxis(fn, 8, dict(zip(fn.arg_names, args))).run()


@pytest.mark.parametrize('change', ['short_x', 'short_y', 'overlap', 'row_overlap', 'duplicate_grid', 'overflow'])
def test_row_reduction_guards_preserve_bounds_and_independence(tensors, change):
    args = list(arguments(tensors))
    recipe = proof(args)
    grid = (3, )
    if change == 'short_x': args[0] = tensors(65536, width=4, count=384)
    elif change == 'short_y': args[1] = tensors(1048576, width=4, count=384)
    elif change == 'overlap': args[1] = tensors(65540, width=4, count=17 * 401)
    elif change == 'row_overlap': args[4] = 128
    elif change == 'duplicate_grid': grid = (3, 2)
    else: args[5] = 2**31 - 1
    assert adapt_tile(recipe, tuple(args), {(8, ): 8, (9, ): 128}, grid, 0, compile_mode='simd') is None
