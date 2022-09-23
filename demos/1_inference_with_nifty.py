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
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

# # Non-linear tomography
#
# The signal is a sigmoid-normal distributed field.
# The data is the field integrated along lines of sight that are
# randomly (set mode=0) or radially (mode=1) distributed

import sys
import os
import numpy as np
import nifty8 as ift
# %matplotlib inline
ift.random.push_sseq_from_seed(27)

""
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    master = comm.Get_rank() == 0
except ImportError:
    comm = None
    master = True


def radial_los(n_los):
    starts = list(ift.random.current_rng().random((n_los, 2)).T)
    ends = list(0.5 + 0*ift.random.current_rng().random((n_los, 2)).T)
    return starts, ends


output_directory="1_inference_with_nifty_results"
os.makedirs(output_directory, exist_ok=True)

position_space = ift.RGSpace([128, 128])

correlated_field = ift.SimpleCorrelatedField(
    position_space,
    offset_mean=0,
    offset_std=(1e-3, 1e-6),
    fluctuations=(1., 0.8),
    loglogavgslope=(-3., 1),
    flexibility=(2, 1.),
    asperity=(0.5, 0.4))
pspec = correlated_field.power_spectrum

signal = ift.sigmoid(correlated_field)

LOS_starts, LOS_ends = radial_los(100)
R = ift.LOSResponse(position_space, starts=LOS_starts, ends=LOS_ends)
signal_response = R @ signal

""
data_space = R.target
noise = .001
N = ift.ScalingOperator(data_space, noise, float)

""
mock_position = ift.from_random(signal_response.domain, 'normal')
data = signal_response(mock_position) + N.draw_sample()

plot = ift.Plot()
plot.add(signal(mock_position), title='Ground Truth', vmin=0, vmax=1)
plot.add(R.adjoint_times(data), title='Data')
plot.add([pspec.force(mock_position)], title='Power Spectrum')
plot.output(ny=1, nx=3, xsize=24, ysize=6)

""
ic_sampling = ift.AbsDeltaEnergyController(#name="Sampling (linear)",
                                           deltaE=0.05, iteration_limit=100)
ic_newton = ift.AbsDeltaEnergyController(name='Newton',
                                         deltaE=0.5,
                                         convergence_level=2, iteration_limit=35)
ic_sampling_nl = ift.AbsDeltaEnergyController(name='Sampling (nonlin)',
                                              deltaE=0.5, iteration_limit=15,
                                              convergence_level=2)
minimizer = ift.NewtonCG(ic_newton)
minimizer_sampling = ift.NewtonCG(ic_sampling_nl)

""
likelihood_energy = (ift.GaussianEnergy(data, inverse_covariance=N.inverse) @
                     signal_response)


# +
plot_directory = os.path.join(output_directory, "plots")
os.makedirs(plot_directory, exist_ok=True)

def inspect_callback(samples, iglobal):
    plot = ift.Plot()
    mean, var = samples.sample_stat(signal)
    plot.add(mean, title="Posterior Mean", vmin=0, vmax=1)
    plot.add(signal(mock_position), title="Ground truth", vmin=0, vmax=1)
    plot.add((mean - signal(mock_position)).abs()/var.sqrt(), title="abs(posterior mean - ground truth) / posterior std", vmin=0, vmax=4)
    plot.add(var.sqrt(), title="Posterior Standard Deviation", vmin=0)

    nsamples = samples.n_samples
    logspec = pspec.log()
    plot.add(list(samples.iterator(pspec)) +
             [pspec.force(mock_position), samples.average(logspec).exp()],
             title="Sampled Posterior Power Spectrum",
             linewidth=[1.]*nsamples + [3., 3.],
             label=[None]*nsamples + ['Ground truth', 'Posterior mean'])
    if master:
        plot.output(ny=3, nx=2, xsize=20, ysize=16, name=os.path.join(plot_directory, f"{iglobal}.png"))


# -

""
n_iterations = 6
n_samples = lambda iiter: 10 if iiter < 5 else 20
samples = ift.optimize_kl(likelihood_energy, n_iterations, n_samples,
                          minimizer, ic_sampling, minimizer_sampling,
                          export_operator_outputs={"signal": signal},
                          inspect_callback=inspect_callback,
                          output_directory="1_inference_with_nifty_results",
                          comm=comm)

""
print(ift.ResidualSampleList.load("1_inference_with_nifty_results/pickle/last", comm=comm))

""

