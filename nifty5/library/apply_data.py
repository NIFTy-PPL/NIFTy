def ApplyData(data, var, model_data):
    # TODO This is rather confusing. Delete that eventually.
    from ..operators.diagonal_operator import DiagonalOperator
    from ..models.constant import Constant
    from ..sugar import sqrt
    sqrt_n = DiagonalOperator(sqrt(var))
    data = Constant(model_data.position, data)
    return sqrt_n.inverse(model_data - data)
