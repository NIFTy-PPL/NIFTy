#!/usr/bin/env python3

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
# Copyright(C) 2013-2021 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

############################################################
# Density estimation
#
# Compute a density estimate for a log-normal process measured by a
# Poissonian likelihood.
#
# Demo takes a while to compute
#############################################################

import numpy as np

import nifty7 as ift

if __name__ == "__main__":
    filename = "getting_started_density_{}.png"
    ift.random.push_sseq_from_seed(42)

    # Set up signal domain
    npix1 = 128
    npix2 = 128
    sp1 = ift.RGSpace(npix1)
    sp2 = ift.RGSpace(npix2)
    position_space = ift.DomainTuple.make((sp1, sp2))

    signal, ops = ift.density_estimator(position_space)
    correlated_field = ops["correlated_field"]

    data_space = signal.target

    # Generate mock signal and data
    rng = ift.random.current_rng()
    mock_position = ift.from_random(signal.domain, "normal")
    data = ift.Field.from_raw(data_space, rng.poisson(signal(mock_position).val))

    # Rejoin domains for plotting
    plotting_domain = ift.DomainTuple.make(ift.RGSpace((npix1, npix2)))
    plotting_domain_expanded = ift.DomainTuple.make(ift.RGSpace((2 * npix1, 2 * npix2)))

    plot = ift.Plot()
    plot.add(
        ift.Field.from_raw(
            plotting_domain_expanded, ift.exp(correlated_field(mock_position)).val
        ),
        title="Pre-Slicing Truth",
    )
    plot.add(
        ift.Field.from_raw(plotting_domain, signal(mock_position).val),
        title="Ground Truth",
    )
    plot.add(ift.Field.from_raw(plotting_domain, data.val), title="Data")
    plot.output(ny=1, nx=3, xsize=10, ysize=3, name=filename.format("setup"))
    print("Setup saved as", filename.format("setup"))

    # Minimization parameters
    ic_sampling = ift.AbsDeltaEnergyController(
        name="Sampling", deltaE=0.01, iteration_limit=100
    )
    ic_newton = ift.AbsDeltaEnergyController(
        name="Newton", deltaE=0.01, iteration_limit=35
    )
    ic_sampling.enable_logging()
    ic_newton.enable_logging()
    minimizer = ift.NewtonCG(ic_newton, enable_logging=True)

    # Number of samples used to estimate the KL
    n_samples = 5

    # Set up likelihood energy and information Hamiltonian
    likelihood_energy = ift.PoissonianEnergy(data) @ signal
    ham = ift.StandardHamiltonian(likelihood_energy, ic_sampling)

    # Start minimization
    initial_mean = ift.MultiField.full(ham.domain, 0.)
    mean = initial_mean

    for i in range(5):
        # Draw new samples and minimize KL
        kl = ift.MetricGaussianKL(mean, ham, n_samples, True)
        kl, convergence = minimizer(kl)
        mean = kl.position

        # Plot current reconstruction
        plot = ift.Plot()
        plot.add(
            ift.Field.from_raw(
                plotting_domain_expanded, ift.exp(correlated_field(mock_position)).val
            ),
            title="Ground truth",
        )
        plot.add(
            ift.Field.from_raw(plotting_domain, signal(mock_position).val),
            title="Ground truth",
        )
        plot.add(
            ift.Field.from_raw(plotting_domain, signal(kl.position).val),
            title="Reconstruction",
        )
        plot.add(
            (ic_newton.history, ic_sampling.history, minimizer.inversion_history),
            label=["kl", "Sampling", "Newton inversion"],
            title="Cumulative energies",
            s=[None, None, 1],
            alpha=[None, 0.2, None],
        )
        plot.output(
            nx=3, ny=2, ysize=10, xsize=15, name=filename.format(f"loop_{i:02d}")
        )

    # Done, draw posterior samples
    sc = ift.StatCalculator()
    sc_unsliced = ift.StatCalculator()
    for sample in kl.samples:
        sc.add(signal(sample + kl.position))
        sc_unsliced.add(ift.exp(correlated_field(sample + kl.position)))

    # Plotting
    plot = ift.Plot()
    plot.add(ift.Field.from_raw(plotting_domain, sc.mean.val), title="Posterior Mean")
    plot.add(
        ift.Field.from_raw(plotting_domain, ift.sqrt(sc.var).val),
        title="Posterior Standard Deviation",
    )
    plot.add(
        ift.Field.from_raw(plotting_domain_expanded, sc_unsliced.mean.val),
        title="Posterior Unsliced Mean",
    )
    plot.add(
        ift.Field.from_raw(plotting_domain_expanded, ift.sqrt(sc_unsliced.var).val),
        title="Posterior Unsliced Standard Deviation",
    )
    plot.output(xsize=15, ysize=15, name=filename.format("results"))
    print("Saved results as", filename.format("results"))
