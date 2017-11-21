from .. import Field, exp
from ..utilities import memo
from ..operators.smoothness_operator import SmoothnessOperator
from ..sugar import generate_posterior_sample
from .nonlinear_power_curvature import NonlinearPowerCurvature
from .response_operators import LinearizedPowerResponse
from ..minimization.energy import Energy
from ..operators.inversion_enabler import InversionEnabler


class NonlinearPowerEnergy(Energy):
    """The Energy of the power spectrum according to the critical filter.

    It describes the energy of the logarithmic amplitudes of the power spectrum of
    a Gaussian random field under reconstruction uncertainty with smoothness and
    inverse gamma prior. It is used to infer the correlation structure of a correlated signal.

    Parameters
    ----------
    position : Field,
        The current position of this energy.
    m : Field,
        The map whichs power spectrum has to be inferred
    D : EndomorphicOperator,
        The curvature of the Gaussian encoding the posterior covariance.
        If not specified, the map is assumed to be no reconstruction.
        default : None
    sigma : float
        The parameter of the smoothness prior.
        default : ??? None? ???????
    samples : integer
        Number of samples used for the estimation of the uncertainty corrections.
        default : 3
    """

    def __init__(self, position, d, N, m, D, FFT, Instrument, nonlinearity, Projection,
                 sigma=0., samples=3, sample_list=None, inverter=None):
        super(NonlinearPowerEnergy, self).__init__(position=position.copy())
        dummy = self.position.norm()
        self.m = m
        self.D = D
        self.d = d
        self.N = N
        self.sigma = sigma
        self.T = SmoothnessOperator(domain=self.position.domain[0], strength=self.sigma,
                                        logarithmic=True)
        self.samples = samples
        self.FFT = FFT
        self.Instrument = Instrument
        self.nonlinearity = nonlinearity
        self.Projection = Projection
        self.LinearizedResponse = LinearizedPowerResponse(self.Instrument, self.nonlinearity,
                                                          self.FFT, self.Projection, self.position, self.m)

        self.power = self.Projection.adjoint_times(
            exp(0.5 * self.position))
        if sample_list is None:
            sample_list = []
            if samples is None:
                sample_list.append(self.m)
            else:
                for i in range(samples):
                    sample = generate_posterior_sample(m, D)
                    sample = FFT(Field(FFT.domain, val=(
                        FFT.adjoint_times(sample).val)))
                    sample_list.append(sample)
        self.sample_list = sample_list
        self.inverter = inverter

    def at(self, position):
        return self.__class__(position, self.d, self.N, self.m,
                              self.D, self.FFT, self.Instrument, self.nonlinearity,
                              self.Projection, sigma=self.sigma,
                              sample_list=self.sample_list,
                              samples=self.samples, inverter=self.inverter)

    @property
    @memo
    def value(self):
        likelihood = 0.
        for sample in self.sample_list:
            likelihood += self._likelihood(sample)
        return 0.5 * self.position.vdot(self.T(self.position)) + likelihood / float(len(self.sample_list))

    def _likelihood(self, m):
        residual = self.d - \
            self.Instrument(self.nonlinearity(
                self.FFT.adjoint_times(self.power * m)))
        energy = 0.5 * residual.vdot(self.N.inverse_times(residual))
        return energy.real

    @property
    @memo
    def gradient(self):
        likelihood_gradient = Field(self.position.domain, val=0.)
        for sample in self.sample_list:
            likelihood_gradient += self._likelihood_gradient(sample)
        return -likelihood_gradient / float(len(self.sample_list)) + self.T(self.position)

    def _likelihood_gradient(self, m):
        residual = self.d - \
            self.Instrument(self.nonlinearity(
                self.FFT.adjoint_times(self.power * m)))
        LinearizedResponse = LinearizedPowerResponse(self.Instrument, self.nonlinearity,
                                                     self.FFT, self.Projection, self.position, m)
        gradient = LinearizedResponse.adjoint_times(
            self.N.inverse_times(residual))
        return gradient

    @property
    @memo
    def curvature(self):
        curvature = NonlinearPowerCurvature(self.position, self.FFT, self.Instrument, self.nonlinearity,
                                            self.Projection, self.N, self.T, self.sample_list, inverter=self.inverter)
        return InversionEnabler(curvature, inverter=self.inverter)
