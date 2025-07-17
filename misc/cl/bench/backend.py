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
# Copyright(C) 2025 LambdaFields GmbH

import cProfile
import io
import pstats

import nifty.cl as ift
import numpy as np

position_space = ift.RGSpace([128, 128])
args = {
    "offset_mean": 0,
    "offset_std": (1e-3, 1e-6),
    "fluctuations": (1.0, 0.8),
    "loglogavgslope": (-3.0, 1),
    "flexibility": (2, 1.0),
    "asperity": (0.5, 0.4),
}
correlated_field = ift.SimpleCorrelatedField(position_space, **args)
pspec = correlated_field.power_spectrum
signal = ift.sigmoid(correlated_field)
R = ift.GeometryRemover(signal.target)
signal_response = R(signal)
data_space = R.target
noise = 0.001
N = ift.ScalingOperator(data_space, noise, np.float64)
mock_position = ift.from_random(signal_response.domain, "normal")
data = signal_response(mock_position) + N.draw_sample()

if False:
    ic_sampling = ift.AbsDeltaEnergyController(
        name="Sampling (linear)", deltaE=0.05, iteration_limit=100
    )
    ic_newton = ift.AbsDeltaEnergyController(
        name="Newton", deltaE=0.5, convergence_level=2, iteration_limit=35
    )
    ic_sampling_nl = ift.AbsDeltaEnergyController(
        name="Sampling (nonlin)", deltaE=0.5, iteration_limit=15, convergence_level=2
    )
    minimizer = ift.NewtonCG(ic_newton)
    minimizer_sampling = ift.NewtonCG(ic_sampling_nl)
else:
    ic_sampling = ift.AbsDeltaEnergyController(
        name="Sampling (linear)", deltaE=0.05, iteration_limit=10
    )
    ic_newton = ift.AbsDeltaEnergyController(
        name="Newton", deltaE=0.5, convergence_level=2, iteration_limit=3
    )
    minimizer = ift.NewtonCG(ic_newton)
    minimizer_sampling = None
likelihood_energy = (
    ift.GaussianEnergy(data, inverse_covariance=N.inverse) @ signal_response
)
n_iterations = 1
n_samples = 4
pr = cProfile.Profile()
pr.enable()
ift.optimize_kl(
    likelihood_energy,
    n_iterations,
    n_samples,
    minimizer,
    ic_sampling,
    nonlinear_sampling_minimizer=minimizer_sampling,
)
pr.disable()

s = io.StringIO()
sortby = 'tottime'
# sortby = 'cumtime'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats(10)
print(s.getvalue())


# Baseline:
#       6756040 function calls (6633319 primitive calls) in 3.095 seconds

# Ordered by: internal time
# List reduced from 694 to 10 due to restriction <10>
#
#  ncalls  tottime  percall  cumtime  percall filename:lineno(function)
# 1961640    0.284    0.000    0.325    0.000 {built-in method builtins.isinstance}
#   38224    0.167    0.000    0.585    0.000 /mnt/nifty/cl/any_array.py:641(__array_ufunc__)
#   10835    0.112    0.000    0.319    0.000 /mnt/nifty/cl/operators/chain_operator.py:43(simplify)
#  115082    0.112    0.000    0.261    0.000 /mnt/nifty/cl/any_array.py:90(__init__)
#   45458    0.098    0.000    0.198    0.000 /mnt/nifty/cl/any_array.py:563(_unify_device_ids_and_get_val)
#     845    0.088    0.000    0.088    0.000 {built-in method scipy.fft._pocketfft.pypocketfft.c2c}
#  126696    0.079    0.000    0.163    0.000 /usr/local/lib/python3.13/site-packages/numpy/_core/numeric.py:1964(isscalar)
#   44594    0.068    0.000    0.170    0.000 /mnt/nifty/cl/field.py:51(__init__)
#   45458    0.057    0.000    0.097    0.000 /mnt/nifty/cl/any_array.py:609(_check_responsibility)
#  115082    0.056    0.000    0.084    0.000 /mnt/nifty/cl/any_array.py:85(__new__)
