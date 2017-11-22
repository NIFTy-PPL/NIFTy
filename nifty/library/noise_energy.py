from .. import Field, exp
from ..operators.diagonal_operator import DiagonalOperator
from ..sugar import generate_posterior_sample
from ..minimization.energy import Energy
from ..utilities import memo


class NoiseEnergy(Energy):
    def __init__(self, position, d, m, D, t, FFT, Instrument, nonlinearity, alpha, q, Projection,
                 samples=3, sample_list=None, inverter=None):
        super(NoiseEnergy, self).__init__(position=position.copy())
        dummy = self.position.norm()
        self.m = m
        self.D = D
        self.d = d
        self.N = DiagonalOperator(diagonal=exp(self.position))
        self.t = t
        self.samples = samples
        self.FFT = FFT
        self.Instrument = Instrument
        self.nonlinearity = nonlinearity

        self.alpha = alpha
        self.q = q
        self.Projection = Projection
        self.power = self.Projection.adjoint_times(exp(0.5 * self.t))
        self.one = Field(self.position.domain, val=1.)
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
        return self.__class__(position, self.d, self.m,
                              self.D, self.t, self.FFT, self.Instrument, self.nonlinearity,
                              self.alpha,
                              self.q,
                              self.Projection,
                              sample_list=self.sample_list,
                              samples=self.samples, inverter=self.inverter)

    @property
    @memo
    def value(self):
        likelihood = 0.
        for sample in self.sample_list:
            likelihood += self._likelihood(sample)
        return ((likelihood / float(len(self.sample_list))) + 0.5 * self.one.vdot(self.position)
                + (self.alpha - self.one).vdot(self.position) + self.q.vdot(exp(-self.position)))

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
        return (likelihood_gradient / float(len(self.sample_list))
                + 0.5 * self.one + (self.alpha - self.one) - self.q * (exp(-self.position)))

    def _likelihood_gradient(self, m):
        residual = self.d - \
            self.Instrument(self.nonlinearity(
                self.FFT.adjoint_times(self.power * m)))
        gradient = -  0.5 * \
            self.N.inverse_times(residual.conjugate() * residual)
        return gradient

    @property
    @memo
    def curvature(self):
        pass
