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
# Copyright(C) 2019-2020 Max-Planck-Society

import numpy as np
from matplotlib import pyplot as plt

import nifty7 as ift


def get_multi_mean(field):
    mean = 0
    for key in field.keys():
        mean += field[key].val.mean()*field[key].size
    return mean/field.size


if __name__ == '__main__':
    comm, _, _, master = ift.utilities.get_MPI_params()
    ift.random.push_sseq_from_seed(42)

    N_chains = 4

    position_space = ift.RGSpace([128])

    cfmaker = ift.CorrelatedFieldMaker.make(
        offset_mean=1.,  # 0.
        offset_std_mean=1e-3,  # 1e-3
        offset_std_std=1e-6,  # 1e-6
        prefix='')

    fluctuations_dict = {
        # Amplitude of field fluctuations
        'fluctuations_mean': 1.0,  # 1.0
        'fluctuations_stddev': .5,  # 1e-2

        # Exponent of power law power spectrum component
        'loglogavgslope_mean': -1.5,  # -3.0
        'loglogavgslope_stddev': 0.2,  #  0.5

        # Amplitude of integrated Wiener process power spectrum component
        'flexibility_mean': 0.5,  # 1.0
        'flexibility_stddev': .25,  # 0.5

        # How ragged the integrated Wiener process component is
        'asperity_mean': 0.5,  # 0.1
        'asperity_stddev': 0.25  # 0.5
    }
    cfmaker.add_fluctuations(position_space, **fluctuations_dict)

    correlated_field = cfmaker.finalize()
    A = cfmaker.amplitude
    # Apply a nonlinearity
    signal = correlated_field.exp()

    # Build the line-of-sight response and define signal response
    Mask = np.ones(position_space.shape)
    Mask[50:80] = 0
    GR = ift.GeometryRemover(signal.target)
    Mask = ift.makeField(GR.target, Mask)
    M = ift.makeOp(Mask)
    R = M @ GR
    signal_response = R(signal)

    # Specify noise
    data_space = R.target
    noise = 10.
    N = ift.ScalingOperator(data_space, noise)

    # Generate mock signal and data
    mock_position = ift.from_random(signal_response.domain, 'normal')
    data = signal_response(mock_position) + M(
        N.draw_sample_with_dtype(dtype=np.float64))

    if master:
        plt.cla()
        plt.figure('result')
        plt.plot(data.val, 'kx', label='data')
        plt.plot(signal(mock_position).val, 'r-', label='ground truth')
        plt.pause(0.01)

    # Set up likelihood and information Hamiltonian
    likelihood = (ift.GaussianEnergy(mean=data, inverse_covariance=N.inverse)
                  @ signal_response)
    H = ift.StandardHamiltonian(likelihood)

    initial_positions = []
    with ift.random.Context(43):
        for i in range(N_chains):
            initial_positions.append(ift.from_random(H.domain))

    HMC = ift.HMC_Sampler(H,
                          initial_positions,
                          steplength=0.03,
                          chains=N_chains,
                          comm=comm)

    for i in range(5):
        HMC.warmup(50)

    ESS_mean = []
    R_hat_mean = []
    for i in range(50):
        HMC.sample(10)

        # Plotting and diagnostics
        ESS = HMC.ESS
        R_Hat = HMC.R_hat
        ESS_mean.append(get_multi_mean(ESS))
        R_hat_mean.append(get_multi_mean(R_Hat) - 1)

        mean_result, var_result = HMC.estimate_quantity(signal)
        mean_power, var_power = HMC.estimate_quantity(A.force)

        if not master:
            continue

        plt.clf()
        fig, axes = plt.subplots(2, 2, num='results', figsize=(12, 8))
        axes[0, 0].plot(data.val, 'kx', label='data')
        axes[0, 0].plot(signal(mock_position).val, 'r-', label='ground truth')
        axes[0, 0].plot(mean_result.val, 'k-', label='sample mean')
        axes[0, 0].fill_between(np.arange(128),
                                mean_result.val - var_result.val**0.5,
                                mean_result.val + var_result.val**0.5,
                                color='k',
                                alpha=0.3)

        axes[0, 1].plot(A.force(mock_position).val, 'r-', label='ground truth')
        axes[0, 1].plot(mean_power.val, 'k-', label='sample mean')
        for i in range(10):
            axes[0, 1].plot(A.force(
                HMC._local_chains[0].samples[np.random.randint(
                    0, len(HMC._local_chains[0].samples))]).val,
                            'k-',
                            alpha=0.3,
                            linewidth=0.5)
            axes[0, 0].plot(signal.force(
                HMC._local_chains[0].samples[np.random.randint(
                    0, len(HMC._local_chains[0].samples))]).val,
                            'k-',
                            alpha=0.3,
                            linewidth=0.5)

        axes[0, 1].set_yscale('log')
        axes[0, 1].set_xscale('log')

        axes[0, 0].set_title('reconstruction')
        axes[0, 1].set_title('amplitude spectrum')
        axes[0, 0].legend()
        axes[0, 1].legend()

        axes[1, 0].plot(ESS_mean, label='mean ESS ')

        axes[1, 1].plot(R_hat_mean, label='mean R_hat - 1')
        axes[1, 1].set_yscale('log')

        axes[1, 0].legend()
        axes[1, 1].legend()
        plt.pause(0.01)
