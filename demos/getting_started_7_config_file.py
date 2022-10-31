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
# Copyright(C) 2013-2022 Max-Planck-Society
# Copyright(C) 2022 Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

############################################################
# Similar to getting_started_3.py but with config files
#############################################################

import sys

import nifty8 as ift


def sky(
    *,
    npix,
    vol,
    offset_mean,
    offset_std_mean,
    offset_std_std,
    fluctuations_mean,
    fluctuations_std,
    loglogavgslope_mean,
    loglogavgslope_std,
    flexibility_mean,
    flexibility_std,
    asperity_mean,
    asperity_std
):
    position_space = ift.RGSpace(npix, vol / npix)
    correlated_field = ift.SimpleCorrelatedField(
        position_space,
        offset_mean,
        offset_std=(offset_std_mean, offset_std_std),
        fluctuations=(fluctuations_mean, fluctuations_std),
        flexibility=(flexibility_mean, flexibility_std),
        asperity=(asperity_mean, asperity_std),
        loglogavgslope=(loglogavgslope_mean, loglogavgslope_std),
    )
    return correlated_field.sigmoid()


def lh(*, sky, noise_var):
    noise_var = float(noise_var)
    R, N, d = mock_RNd(
        sky, noise_var
    )  # This would be replaced by a data loading routine
    return ift.GaussianEnergy(d, inverse_covariance=N.inverse) @ R @ sky


def trans(*, sky_before, sky_after):
    # FIXME Write a transition that e.g. transitions from a lower to a higher resolved grid
    return lambda sample_list: ift.full(sky_after.domain, 0.)


builder_dct = {"lh0": lh, "lh1": lh, "sky0": sky, "sky1": sky, "trans01": trans}


def main():
    _, cfg_file = sys.argv
    cfg = ift.OptimizeKLConfig.from_file(cfg_file, builder_dct)
    cfg.optimize_kl(
        export_operator_outputs={"sky1": cfg.instantiate_section("sky1")},
    )


def mock_RNd(signal_operator, noise_var):
    sdom = signal_operator.target
    p = 0.2
    with ift.random.Context(42):
        mask = ift.random.current_rng().choice(
            a=[False, True], size=sdom.shape, p=[p, 1 - p]
        )
        R = ift.MaskOperator(ift.makeField(sdom, mask))
        N = ift.ScalingOperator(R.target, noise_var, float)
        mock_position = ift.from_random(signal_operator.domain, "normal")
        d = (R @ signal_operator)(mock_position) + N.draw_sample()
    return R, N, d


if __name__ == "__main__":
    main()
