import numpy as np

from ..domains.power_space import PowerSpace
from ..domains.unstructured_domain import UnstructuredDomain
from ..field import Field
from ..multi.multi_field import MultiField
from ..sugar import makeOp, sqrt


def _ceps_kernel(dof_space, k, a, k0):
    return a**2/(1+(k/(k0*dof_space.bindistances[0]))**2)**2


def make_amplitude_model(s_space, Npixdof, ceps_a, ceps_k, sm, sv, im, iv,
                         keys=['tau', 'phi']):
    '''
    Method for construction of amplitude model
    Computes a smooth power spectrum.
    Output lives in PowerSpace.

    Parameters
    ----------

    Npixdof : #pix in dof_space

    ceps_a, ceps_k0 : Smoothness parameters in ceps_kernel
                        eg. ceps_kernel(k) = (a/(1+(k/k0)**2))**2
                        a = ceps_a,  k0 = ceps_k0

    sm, sv : slope_mean = expected exponent of power law (e.g. -4),
                slope_variance (default=1)

    im, iv : y-intercept_mean, y-intercept_variance  of power_slope
    '''
    from ..operators import (ExpTransform, QHTOperator, SlopeOperator,
                             SymmetrizingOperator)
    from ..models import Variable, Constant, PointwiseExponential

    h_space = s_space.get_default_codomain()
    p_space = PowerSpace(h_space)
    exp_transform = ExpTransform(p_space, Npixdof)
    logk_space = exp_transform.domain[0]
    dof_space = logk_space.get_default_codomain()
    param_space = UnstructuredDomain(2)
    qht = QHTOperator(dof_space, logk_space)
    sym = SymmetrizingOperator(logk_space)

    phi_mean = np.array([sm, im])
    phi_sig = np.array([sv, iv])

    slope = SlopeOperator(param_space, logk_space, phi_sig)
    norm_phi_mean = Field.from_global_data(param_space, phi_mean/phi_sig)

    fields = {keys[0]: Field.from_random('normal', dof_space),
              keys[1]: Field.from_random('normal', param_space)}

    position = MultiField(fields)

    dof_space = position[keys[0]].domain[0]
    kern = lambda k: _ceps_kernel(dof_space, k, ceps_a, ceps_k)
    cepstrum = create_cepstrum_amplitude_field(dof_space, kern)

    ceps = makeOp(sqrt(cepstrum))
    smooth_op = sym * qht * ceps
    smooth_spec = smooth_op(Variable(position)[keys[0]])
    phi = Variable(position)[keys[1]] + Constant(position, norm_phi_mean)
    linear_spec = slope(phi)
    loglog_spec = smooth_spec + linear_spec
    xlog_ampl = PointwiseExponential(0.5*loglog_spec)

    internals = {'loglog_spec': loglog_spec,
                 'qht': qht,
                 'ceps': ceps,
                 'norm_phi_mean': norm_phi_mean}
    return exp_transform(xlog_ampl), internals


def create_cepstrum_amplitude_field(domain, cepstrum):
    """Creates a ...
    Writes the sum of all modes into the zero-mode.

    Parameters
    ----------
    domain: ???
        ???
    cepstrum: Callable
        ???
    """

    dim = len(domain.shape)
    dist = domain.bindistances
    shape = domain.shape

    # Prepare q_array
    q_array = np.zeros((dim,) + shape)
    if dim == 1:
        ks = domain.get_k_length_array().to_global_data()
        q_array = np.array([ks])
    else:
        for i in range(dim):
            ks = np.minimum(shape[i] - np.arange(shape[i]) +
                            1, np.arange(shape[i])) * dist[i]
            fst_dims = (1,) * i
            lst_dims = (1,) * (dim - i - 1)
            q_array[i] += ks.reshape(fst_dims + (shape[i],) + lst_dims)

    # Fill cepstrum field (all non-zero modes)
    no_zero_modes = (slice(1, None),) * dim
    ks = q_array[(slice(None),) + no_zero_modes]
    cepstrum_field = np.zeros(shape)
    cepstrum_field[no_zero_modes] = cepstrum(ks)

    # Fill cepstrum field (zero-mode subspaces)
    for i in range(dim):
        # Prepare indices
        fst_dims = (slice(None),) * i
        lst_dims = (slice(None),) * (dim - i - 1)
        sl = fst_dims + (slice(1, None),) + lst_dims
        sl2 = fst_dims + (0,) + lst_dims

        # Do summation
        cepstrum_field[sl2] = np.sum(cepstrum_field[sl], axis=i)

    return Field.from_global_data(domain, cepstrum_field)
