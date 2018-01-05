from .scaling_operator import ScalingOperator
from .laplace_operator import LaplaceOperator


def SmoothnessOperator(domain, strength=1., logarithmic=True, space=None):
    """An operator measuring the smoothness on an irregular grid with respect
    to some scale.

    This operator applies the irregular LaplaceOperator and its adjoint to some
    Field over a PowerSpace which corresponds to its smoothness and weights the
    result with a scale parameter sigma. It is used in the smoothness prior
    terms of the CriticalPowerEnergy. For this purpose we use free boundary
    conditions in the LaplaceOperator, having no curvature at both ends. In
    addition the first entry is ignored as well, corresponding to the overall
    mean of the map. The mean is therefore not considered in the smoothness
    prior.


    Parameters
    ----------
    strength: nonnegative float
        Specifies the strength of the SmoothnessOperator
    logarithmic : boolean
        Whether smoothness is calculated on a logarithmic scale or linear scale
        default : True
    """
    if strength < 0:
        raise ValueError("ERROR: strength must be nonnegative.")
    if strength == 0.:
        return ScalingOperator(0., domain)
    laplace = LaplaceOperator(domain, logarithmic=logarithmic, space=space)
    return (strength**2)*laplace.adjoint*laplace
