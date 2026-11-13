# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Shared arithmetic only; guard versions retain their own domains and budgets."""


def affine_uniform(value):
    return not any(value.coefficients)


def affine_add(lhs, rhs, sign, result_type):
    # Preserve the legacy zip semantics and the caller's concrete result type.
    return result_type(lhs.offset + sign * rhs.offset,
                       tuple(a + sign * b for a, b in zip(lhs.coefficients, rhs.coefficients)))


def affine_scale(value, factor, result_type):
    return result_type(value.offset * factor, tuple(c * factor for c in value.coefficients))
