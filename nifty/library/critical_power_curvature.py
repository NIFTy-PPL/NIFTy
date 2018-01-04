from ..operators import InversionEnabler, DiagonalOperator


def CriticalPowerCurvature(theta, T, inverter):
    theta = DiagonalOperator(theta)
    return InversionEnabler(T+theta, inverter, theta.inverse_times)
