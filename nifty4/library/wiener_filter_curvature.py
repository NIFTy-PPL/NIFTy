# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from ..operators import EndomorphicOperator, InversionEnabler
from ..field import Field, sqrt
from ..sugar import power_analyze, power_synthesize


class WienerFilterCurvature(EndomorphicOperator):
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
        super(WienerFilterCurvature, self).__init__()
        self.R = R
        self.N = N
        self.S = S
        op = R.adjoint*N.inverse*R + S.inverse
        self._op = InversionEnabler(op, inverter, S.times)

    @property
    def domain(self):
        return self._op.domain

    @property
    def capability(self):
        return self._op.capability

    def apply(self, x, mode):
        return self._op.apply(x, mode)

    def generate_posterior_sample(self):
        """ Generates a posterior sample from a Gaussian distribution with
        given mean and covariance.

        This method generates samples by setting up the observation and
        reconstruction of a mock signal in order to obtain residuals of the
        right correlation which are added to the given mean.

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
                                       domain=self.N.domain, dtype=noise.dtype)
        mock_noise *= sqrt(noise)

        mock_data = self.R(mock_signal) + mock_noise

        mock_j = self.R.adjoint_times(self.N.inverse_times(mock_data))
        mock_m = self.inverse_times(mock_j)
        return mock_signal - mock_m

    def generate_posterior_sample2(self):
        power = self.S.diagonal()
        mock_signal = Field.from_random(random_type="normal",
                                        domain=self.S.domain, dtype=power.dtype)
        mock_signal *= sqrt(power)

        noise = self.N.diagonal()
        mock_noise = Field.from_random(random_type="normal",
                                       domain=self.N.domain, dtype=noise.dtype)
        mock_noise *= sqrt(noise)

        mock_data = self.R(mock_signal) + mock_noise

        mock_j = self.R.adjoint_times(self.N.inverse_times(mock_data))
        mock_m = self.inverse_times(mock_j)
        return mock_signal - mock_m
