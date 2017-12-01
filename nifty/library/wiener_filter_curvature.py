from ..operators import EndomorphicOperator, InversionEnabler
from ..field import Field, sqrt
from ..sugar import power_analyze, power_synthesize


class WienerFilterCurvature(InversionEnabler, EndomorphicOperator):
    """The curvature of the WienerFilterEnergy.

    This operator implements the second derivative of the
    WienerFilterEnergy used in some minimization algorithms or
    for error estimates of the posterior maps. It is the
    inverse of the propagator operator.

    Parameters
    ----------
    R: LinearOperator,
       The response operator of the Wiener filter measurement.
    N: EndomorphicOperator
       The noise covariance.
    S: DiagonalOperator,
       The prior signal covariance
    """

    def __init__(self, R, N, S, inverter):
        EndomorphicOperator.__init__(self)
        InversionEnabler.__init__(self, inverter, S.times)
        self.R = R
        self.N = N
        self.S = S

    @property
    def domain(self):
        return self.S.domain

    @property
    def self_adjoint(self):
        return True

    @property
    def unitary(self):
        return False

    def _times(self, x):
        res = self.R.adjoint_times(self.N.inverse_times(self.R(x)))
        res += self.S.inverse_times(x)
        return res

    def generate_posterior_sample(self, mean):
        """ Generates a posterior sample from a Gaussian distribution with
        given mean and covariance.

        This method generates samples by setting up the observation and
        reconstruction of a mock signal in order to obtain residuals of the
        right correlation which are added to the given mean.

        Parameters
        ----------
        mean : Field
            the mean of the posterior Gaussian distribution

        Returns
        -------
        sample : Field
            Returns the a sample from the Gaussian of given mean and
            covariance.
        """

        power = sqrt(power_analyze(self.S.diagonal()))
        mock_signal = power_synthesize(power, real_signal=True)

        noise = self.N.diagonal().weight(-1)

        mock_noise = Field.from_random(random_type="normal",
                                       domain=self.N.domain,
                                       dtype=noise.dtype.type)
        mock_noise *= sqrt(noise)

        mock_data = self.R(mock_signal) + mock_noise

        mock_j = self.R.adjoint_times(self.N.inverse_times(mock_data))
        mock_m = self.inverse_times(mock_j)
        sample = mock_signal - mock_m + mean
        return sample
