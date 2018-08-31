from ..operators.energy_operators import InverseGammaLikelihood
from ..operators.scaling_operator import ScalingOperator

def make_adjust_variances(a,xi,position,samples=[],scaling=None):
    """ Creates a Likelihood for constant likelihood optimizations.
    
    Constructs a Likelihood to solve constant likelihood optimizations of the form
        phi = a * xi
    under the constraint that phi remains constant.
    
    Parameters
    ----------
    a : Operator
        Operator which gives the amplitude when evaluated at a position
    xi : Operator
        Operator which gives the excitation when evaluated at a position
    postion : Field, MultiField
        Position of the whole problem
    res_samples : Field, MultiField
        Residual samples of the whole Problem
    scaling : Float
        Optional rescaling of the Likelihood

    Returns
    -------
    InverseGammaLikelihood
        A Likelihood that can be used for further minimization
    """

    d = a * xi
    d = (d.conjugate()*d).real
    n = len(samples)
    if n>0:
        d_eval = 0.
        for i in range(n):
            d_eval = d_eval + d(position+samples[i])
        d_eval = d_eval / n
    else:
        d_eval = d(position)

    x = (a.conjugate()*a).real
    if scaling is not None:
        x = ScalingOperator(scaling,x.target)(x)

    return InverseGammaLikelihood(x,d_eval)