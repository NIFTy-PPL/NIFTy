import matplotlib.pyplot as plt
import numpy as np

import nifty5 as ift
np.random.seed(12)


def polynomial(coefficients, sampling_points):
    """Computes values of polynomial whose coefficients are stored in
    coefficients at sampling points. This is a quick version of the
    PolynomialResponse.

    Parameters
    ----------
    coefficients: Model
    sampling_points: Numpy array
    """

    if not (isinstance(coefficients, ift.Model)
            and isinstance(sampling_points, np.ndarray)):
        raise TypeError
    params = coefficients.value.to_global_data()
    out = np.zeros_like(sampling_points)
    for ii in range(len(params)):
        out += params[ii] * sampling_points**ii
    return out


class PolynomialResponse(ift.LinearOperator):
    """Calculates values of a polynomial parameterized by input at sampling points.

    Parameters
    ----------
    domain: UnstructuredDomain
        The domain on which the coefficients of the polynomial are defined.
    sampling_points: Numpy array
        x-values of the sampling points.
    """

    def __init__(self, domain, sampling_points):
        super(PolynomialResponse, self).__init__()
        if not (isinstance(domain, ift.UnstructuredDomain)
                and isinstance(x, np.ndarray)):
            raise TypeError
        self._domain = ift.DomainTuple.make(domain)
        tgt = ift.UnstructuredDomain(sampling_points.shape)
        self._target = ift.DomainTuple.make(tgt)

        sh = (self.target.size, domain.size)
        self._mat = np.empty(sh)
        for d in range(domain.size):
            self._mat.T[d] = sampling_points**d

    def apply(self, x, mode):
        self._check_input(x, mode)
        val = x.to_global_data()
        if mode == self.TIMES:
            # FIXME Use polynomial() here
            out = self._mat.dot(val)
        else:
            # FIXME Can this be optimized?
            out = self._mat.conj().T.dot(val)
        return ift.from_global_data(self._tgt(mode), out)

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES


# Generate some mock data
N_params = 10
N_samples = 100
size = (12,)
x = np.random.random(size) * 10
y = np.sin(x**2) * x**3
var = np.full_like(y, y.var() / 10)
var[-2] *= 4
var[5] /= 2
y[5] -= 0

# Set up minimization problem
p_space = ift.UnstructuredDomain(N_params)
params = ift.Variable(ift.MultiField.from_dict(
    {'params': ift.full(p_space, 0.)}))['params']
R = PolynomialResponse(p_space, x)
ift.extra.consistency_check(R)

d_space = R.target
d = ift.from_global_data(d_space, y)
N = ift.DiagonalOperator(ift.from_global_data(d_space, var))

IC = ift.GradientNormController(tol_abs_gradnorm=1e-8)
H = ift.Hamiltonian(ift.GaussianEnergy(R(params), d, N), IC)
H = H.make_invertible(IC)

# Minimize
minimizer = ift.RelaxedNewton(IC)
H, _ = minimizer(H)

# Draw posterior samples
samples = [H.metric.draw_sample(from_inverse=True) + H.position
           for _ in range(N_samples)]

# Plotting
plt.errorbar(x, y, np.sqrt(var), fmt='ko', label='Data with error bars')
xmin, xmax = x.min(), x.max()
xs = np.linspace(xmin, xmax, 100)

sc = ift.StatCalculator()
for ii in range(len(samples)):
    sc.add(params.at(samples[ii]).value)
    ys = polynomial(params.at(samples[ii]), xs)
    if ii == 0:
        plt.plot(xs, ys, 'k', alpha=.05, label='Posterior samples')
        continue
    plt.plot(xs, ys, 'k', alpha=.05)
ys = polynomial(params.at(H.position), xs)
plt.plot(xs, ys, 'r', linewidth=2., label='Interpolation')
plt.legend()
plt.savefig('fit.png')
plt.close()

# Print parameters
mean = sc.mean.to_global_data()
sigma = np.sqrt(sc.var.to_global_data())
for ii in range(len(mean)):
    print('Coefficient x**{}: {:.2E} +/- {:.2E}'.format(ii, mean[ii],
                                                        sigma[ii]))