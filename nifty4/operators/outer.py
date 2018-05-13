from .diagonal_operator import DiagonalOperator


def OuterOperator(field, row_operator):
    return DiagonalOperator(field) * row_operator
