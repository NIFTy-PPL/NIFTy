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
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.


############################################################
# Wiener filter demo code 
# showing how measurement gaps are filled in 
# 1D (set mode=0), 2D (mode=1), or on the sphere (mode=2)
#############################################################

import nifty5 as ift
import numpy as np


def make_chess_mask(position_space):
    # set up a checkerboard mask for 2D mode
    mask = np.ones(position_space.shape)
    for i in range(4):
        for j in range(4):
            if (i+j) % 2 == 0:
                mask[i*128//4:(i+1)*128//4, j*128//4:(j+1)*128//4] = 0
    return mask


def make_random_mask():
    # set up random mask for spherical mode
    mask = ift.from_random('pm1', position_space)
    mask = (mask+1)/2
    return mask.to_global_data()


def mask_to_nan(mask, field):
    # mask masked pixels for plotting data
    masked_data = field.local_data.copy()
    masked_data[mask.local_data == 0] = np.nan
    return ift.from_local_data(field.domain, masked_data)


if __name__ == '__main__':
    np.random.seed(42)

    # Choose position space of signal and masking in measurement
    mode = 1
    if mode == 0:
        # One dimensional regular grid
        position_space = ift.RGSpace([1024])
        mask = np.ones(position_space.shape)
    elif mode == 1:
        # Two dimensional regular grid with chess mask
        position_space = ift.RGSpace([128, 128])
        mask = make_chess_mask(position_space)
    else:
        # Sphere with half of its locations randomly masked
        position_space = ift.HPSpace(128)
        mask = make_random_mask()

    # specify harmonic space corresponding to signal
    # the signal degree of freedom will live here
    harmonic_space = position_space.get_default_codomain()
 
    # harmonic transform from harmonic space to position space
    HT = ift.HarmonicTransformOperator(harmonic_space, target=position_space)

    # Set correlation structure with a power spectrum and build
    # prior correlation covariance

    # define 1D isotropic power spectrum
    def power_spectrum(k):
        return 100. / (20.+k**3)
    # 1D spectral space in which power spectrum lives
    power_space = ift.PowerSpace(harmonic_space) 
    # its mapping to (higher dimentional) harmonic space
    PD = ift.PowerDistributor(harmonic_space, power_space) 
    # applying the mapping to the 1D spectrum
    prior_correlation_structure = PD(ift.PS_field(power_space, power_spectrum))
    # inserting the result into the diagonal of an harmonic space operator
    S = ift.DiagonalOperator(prior_correlation_structure)
    # S is now the field covariance

    # Build instrument response consisting of a discretization, mask
    # and harmonic transformaion

    # data lives in geometry free space, thus we need to remove geometry
    GR = ift.GeometryRemover(position_space)
    # masking operator, to model that parts of the field where not observed
    mask = ift.Field.from_global_data(position_space, mask)
    Mask = ift.DiagonalOperator(mask)
    # compile response operator out of 
    # - an harmic transform (to get to image space)
    # - the application of the mask
    # - the removal of geometric information
    R = GR(Mask(HT))

    # find out where the data then lives
    data_space = GR.target

    # Set the noise covariance N
    noise = 5.
    N = ift.ScalingOperator(noise, data_space)

    # Create mock data
    MOCK_SIGNAL = S.draw_sample()
    MOCK_NOISE = N.draw_sample()
    data = R(MOCK_SIGNAL) + MOCK_NOISE

    # Build propagator D and information source j
    D_inv = R.adjoint(N.inverse(R)) + S.inverse
    j = R.adjoint_times(N.inverse_times(data))
    # Make propagator invertible
    IC = ift.GradientNormController(iteration_limit=500, tol_abs_gradnorm=1e-3)
    D = ift.InversionEnabler(D_inv, IC, approximation=S.inverse).inverse

    # WIENER FILTER
    # m = D j
    m = D(j)

    # PLOTTING
    rg = isinstance(position_space, ift.RGSpace)
    plot = ift.Plot()
    if rg and len(position_space.shape) == 1:
        plot.add([HT(MOCK_SIGNAL), GR.adjoint(data), HT(m)],
                 label=['Mock signal', 'Data', 'Reconstruction'],
                 alpha=[1, .3, 1])
        plot.add(mask_to_nan(mask, HT(m-MOCK_SIGNAL)), title='Residuals')
        plot.output(nx=2, ny=1, xsize=10, ysize=4, title="getting_started_1")
    else:
        plot.add(HT(MOCK_SIGNAL), title='Mock Signal')
        plot.add(mask_to_nan(mask, (GR(Mask)).adjoint(data)),
                 title='Data')
        plot.add(HT(m), title='Reconstruction')
        plot.add(mask_to_nan(mask, HT(m-MOCK_SIGNAL)), title='Residuals')
        plot.output(nx=2, ny=2, xsize=10, ysize=10, title="getting_started_1")
