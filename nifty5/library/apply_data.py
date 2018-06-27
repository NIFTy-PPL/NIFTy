def ApplyData(data, var, model_data):
    from .. import DiagonalOperator, Constant, sqrt
    sqrt_n = DiagonalOperator(sqrt(var))
    data = Constant(model_data.position, data)
    return sqrt_n.inverse(model_data - data)
