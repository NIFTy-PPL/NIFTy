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

############################################################
# Proof of concept of the evidence lower bound (ELBO)
# for model comparison.
#
# Conceptually the model is simply a 1D version of
# getting_started_3.py.
# The signal is a sigmoid-normal distributed field.
# The data is the field integrated along lines of sight that
# are randomly (set mode=0) or radially (mode=1) distributed,
# and it is drawn from a process with model 1 statistics.
#
# The data is finally fit with both model 1 and model 2.
# The respective evidences (ELBOs) are compared in order to
# establish which model is to be preferred.
#
# Demo takes a while to compute (especially the ELBO part).
#############################################################

import sys

import nifty8 as ift
import numpy as np

ift.random.push_sseq_from_seed(28)


def random_los(n_los):
    starts = list(ift.random.current_rng().random((n_los, 1)).T)
    ends = list(ift.random.current_rng().random((n_los, 1)).T)
    return starts, ends


def radial_los(n_los):
    starts = list(ift.random.current_rng().random((n_los, 1)).T)
    ends = list(0.5 + 0 * ift.random.current_rng().random((n_los, 1)).T)
    return starts, ends


def main():
    # Choose between random line-of-sight response (mode=0) and radial lines
    # of sight (mode=1)
    if len(sys.argv) == 2:
        mode = int(sys.argv[1])
    else:
        mode = 0
    filename = "getting_started_model_comparison_mode_{}_".format(mode) + "{}.png"
    position_space = ift.RGSpace(128)

    #  For a detailed showcase of the effects the parameters
    #  of the CorrelatedField model have on the generated fields,
    #  see 'getting_started_4_CorrelatedFields.ipynb'.

    args_1 = {
        'offset_mean': 0,
        'offset_std': (1e-3, 1e-6),

        # Amplitude of field fluctuations
        'fluctuations': (1., 0.8),  # 1.0, 1e-2

        # Exponent of power law power spectrum component
        'loglogavgslope': (-3., 1),  # -6.0, 1

        # Amplitude of integrated Wiener process power spectrum component
        'flexibility': (2, 1.),  # 1.0, 0.5

        # How ragged the integrated Wiener process component is
        'asperity': (0.5, 0.4)  # 0.1, 0.5
    }

    args_2 = {
        'offset_mean': 0,
        'offset_std': (1e-3, 1e-6),

        # Amplitude of field fluctuations
        'fluctuations': (3., 0.8),  # 1.0, 1e-2

        # Exponent of power law power spectrum component
        'loglogavgslope': (-3., 1),  # -6.0, 1

        # Amplitude of integrated Wiener process power spectrum component
        'flexibility': (1.5, 1.),  # 1.0, 0.5

        # How ragged the integrated Wiener process component is
        'asperity': (2.5, 0.4)  # 0.1, 0.5
    }

    correlated_field_1 = ift.SimpleCorrelatedField(position_space, **args_1)
    pspec_1 = correlated_field_1.power_spectrum

    correlated_field_2 = ift.SimpleCorrelatedField(position_space, **args_2)
    pspec_2 = correlated_field_2.power_spectrum

    # Apply a nonlinearity
    signal_1 = ift.sigmoid(correlated_field_1)
    signal_2 = ift.sigmoid(correlated_field_2)

    # Build the line-of-sight response and define signal response
    LOS_starts, LOS_ends = random_los(100) if mode == 0 else radial_los(100)
    R = ift.LOSResponse(position_space, starts=LOS_starts, ends=LOS_ends)
    signal_response_1 = R(signal_1)
    signal_response_2 = R(signal_2)

    # Specify noise
    data_space = R.target
    noise = .001
    N = ift.ScalingOperator(data_space, noise, np.float64)

    # Generate mock signal and data
    mock_position = ift.from_random(signal_response_1.domain, 'normal')
    data = signal_response_1(mock_position) + N.draw_sample()

    # Minimization parameters
    ic_sampling = ift.AbsDeltaEnergyController(name="Sampling (linear)",
                                               deltaE=0.05, iteration_limit=100)
    ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=0.5,
                                             convergence_level=2, iteration_limit=35)
    ic_sampling_nl = ift.AbsDeltaEnergyController(name='Sampling (nonlin)',
                                                  deltaE=0.5, iteration_limit=15,
                                                  convergence_level=2)
    minimizer = ift.NewtonCG(ic_newton)
    minimizer_sampling = ift.NewtonCG(ic_sampling_nl)

    # Set up likelihood energy
    likelihood_energy_1 = (ift.GaussianEnergy(data=data, inverse_covariance=N.inverse) @
                           signal_response_1)

    likelihood_energy_2 = (ift.GaussianEnergy(data=data, inverse_covariance=N.inverse) @
                           signal_response_2)

    # Minimize KL
    n_iterations = 6
    n_samples = lambda iiter: 10 if iiter < 5 else 20
    samples_1 = ift.optimize_kl(likelihood_energy_1, n_iterations, n_samples,
                                minimizer, ic_sampling, minimizer_sampling,
                                export_operator_outputs={"signal": signal_1},
                                output_directory="getting_started_model_comparison_results/model_1")
    samples_2 = ift.optimize_kl(likelihood_energy_2, n_iterations, n_samples,
                                minimizer, ic_sampling, minimizer_sampling,
                                export_operator_outputs={"signal": signal_2},
                                output_directory="getting_started_model_comparison_results/model_2")

    # Plotting
    filename_res = filename.format("results")
    plot = ift.Plot()
    mean_1, var_1 = samples_1.sample_stat(signal_1)
    mean_2, var_2 = samples_2.sample_stat(signal_2)
    plot.add(data, title="Data")
    plot.add([signal_1(mock_position), mean_1, mean_2], title='Reconstruction',
             label=['Ground truth', 'Posterior Mean 1', 'Posterior Mean 2'])
    plot.add(var_1.sqrt(), title="Posterior Standard Deviation Model 1")
    plot.add(var_2.sqrt(), title="Posterior Standard Deviation Model 2")

    nsamples_1 = samples_1.n_samples
    logspec_1 = pspec_1.log()
    plot.add(list(samples_1.iterator(pspec_1)) +
             [pspec_1.force(mock_position), samples_1.average(logspec_1).exp()],
             title="Sampled Posterior Power Spectrum Model 1",
             linewidth=[1.] * nsamples_1 + [3., 3.],
             label=[None] * nsamples_1 + ['Ground truth', 'Posterior mean'])

    nsamples_2 = samples_2.n_samples
    logspec_2 = pspec_2.log()
    plot.add(list(samples_2.iterator(pspec_2)) +
             [pspec_2.force(mock_position), samples_2.average(logspec_2).exp()],
             title="Sampled Posterior Power Spectrum Model 2",
             linewidth=[1.] * nsamples_2 + [3., 3.],
             label=[None] * nsamples_2 + ['Ground truth (model 2)', 'Posterior mean'])

    plot.output(ny=2, nx=3, xsize=24, ysize=12, name=filename_res)
    ift.logger.info("Saved results as '{}'.".format(filename_res))

    # Compute evidence lower bound
    evidence_1, _ = ift.estimate_evidence_lower_bound(ift.StandardHamiltonian(likelihood_energy_1), samples_1, 100,
                                                      min_lh_eval=1e-3)
    evidence_2, _ = ift.estimate_evidence_lower_bound(ift.StandardHamiltonian(likelihood_energy_2), samples_2, 99,
                                                      min_lh_eval=1e-3)

    log_elbo_difference = evidence_1.average().val - evidence_2.average().val
    ift.logger.info("\n")
    elbo_ratio = np.exp(log_elbo_difference)
    if log_elbo_difference > 0:
        s = f"Model 1 is favored over model 2 by a factor of {elbo_ratio}."
    else:
        s = f"Model 2 is favored over model 1 by a factor of {1/elbo_ratio}."
    ift.logger.info(s)


if __name__ == '__main__':
    main()
