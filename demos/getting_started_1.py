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
# Copyright(C) 2013-2020 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

###############################################################################
# Compute a Wiener filter solution with NIFTy
# Shows how measurement gaps are filled in
# 1D (set mode=0), 2D (mode=1), or on the sphere (mode=2)
###############################################################################

import sys

import numpy as np

import nifty8 as ift


def make_checkerboard_mask(position_space):
    # Checkerboard mask for 2D mode
    mask = np.ones(position_space.shape)
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 0:
                mask[i*128//4:(i + 1)*128//4, j*128//4:(j + 1)*128//4] = 0
    return mask


def make_random_mask(domain):
    # Random mask for spherical mode
    mask = ift.from_random(domain, 'pm1')
    mask = (mask + 1)/2
    return mask.val


def main():
    # Choose space on which the signal field is defined
    if len(sys.argv) == 2:
        mode = int(sys.argv[1])
    else:
        mode = 1

    if mode == 0:
        # One-dimensional regular grid
        position_space = ift.RGSpace([1024])
        mask = np.zeros(position_space.shape)
    elif mode == 1:
        # Two-dimensional regular grid with checkerboard mask
        position_space = ift.RGSpace([128, 128])
        mask = make_checkerboard_mask(position_space)
    else:
        # Sphere with half of its pixels randomly masked
        position_space = ift.HPSpace(128)
        mask = make_random_mask(position_space)

    # Specify harmonic space corresponding to signal
    harmonic_space = position_space.get_default_codomain()

    # Harmonic transform from harmonic space to position space
    HT = ift.HarmonicTransformOperator(harmonic_space, target=position_space)

    # Set prior correlation covariance with a power spectrum leading to
    # homogeneous and isotropic statistics
    def power_spectrum(k):
        return 100./(20. + k**3)

    # 1D spectral space on which the power spectrum is defined
    power_space = ift.PowerSpace(harmonic_space)

    # Mapping to (higher dimensional) harmonic space
    PD = ift.PowerDistributor(harmonic_space, power_space)

    # Apply the mapping
    prior_correlation_structure = PD(ift.PS_field(power_space, power_spectrum))

    # Insert the result into the diagonal of an harmonic space operator
    S = ift.makeOp(prior_correlation_structure, sampling_dtype=float)
    # S is the prior field covariance

    # Build instrument response consisting of a discretization, mask
    # and harmonic transformaion

    # Masking operator to model that parts of the field have not been observed
    mask = ift.Field.from_raw(position_space, mask)
    Mask = ift.MaskOperator(mask)

    # The response operator consists of
    # - a harmonic transform (to get to image space)
    # - the application of the mask
    # - the removal of geometric information
    # The removal of geometric information is included in the MaskOperator
    # it can also be implemented with a GeometryRemover
    # Operators can be composed either with parenthesis
    R = Mask(HT)
    # or with @
    R = Mask @ HT

    data_space = R.target

    # Set the noise covariance N
    noise = 5.
    N = ift.ScalingOperator(data_space, noise, float)

    # Create mock data
    MOCK_SIGNAL = S.draw_sample()
    MOCK_NOISE = N.draw_sample()
    data = R(MOCK_SIGNAL) + MOCK_NOISE

    # Build inverse propagator D and information source j
    D_inv = R.adjoint @ N.inverse @ R + S.inverse
    j = R.adjoint_times(N.inverse_times(data))
    # Make D_inv invertible (via Conjugate Gradient)
    IC = ift.GradientNormController(iteration_limit=500, tol_abs_gradnorm=1e-3)
    D = ift.InversionEnabler(D_inv, IC, approximation=S.inverse).inverse

    # Calculate WIENER FILTER solution
    m = D(j)

    # Plotting
    rg = isinstance(position_space, ift.RGSpace)
    plot = ift.Plot()
    filename = "getting_started_1_mode_{}.png".format(mode)
    if rg and len(position_space.shape) == 1:
        plot.add(
            [HT(MOCK_SIGNAL), Mask.adjoint(data),
             HT(m)],
            label=['Mock signal', 'Data', 'Reconstruction'],
            alpha=[1, .3, 1])
        plot.add(Mask.adjoint(Mask(HT(m - MOCK_SIGNAL))), title='Residuals')
        plot.output(nx=2, ny=1, xsize=10, ysize=4, name=filename)
    else:
        plot.add(HT(MOCK_SIGNAL), title='Mock Signal')
        plot.add(Mask.adjoint(data), title='Data')
        plot.add(HT(m), title='Reconstruction')
        plot.add(Mask.adjoint(Mask(HT(m) - HT(MOCK_SIGNAL))),
                 title='Residuals')
        plot.output(nx=2, ny=2, xsize=10, ysize=10, name=filename)
    print("Saved results as '{}'.".format(filename))


if __name__ == '__main__':
    main()
