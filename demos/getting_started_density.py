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
# Copyright(C) 2013-2020 Max-Planck-Society
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

import sys
import numpy as np
import nifty7 as ift


def density_estimator(
        domain, exposure=1., pad=1., cf_fluctuations=None, cf_azm=None
    ):
    cf_azm_sane_default = (0., (1e-2, 1e-6))
    cf_fluctuations_sane_default = {
        "scale": (0.5, 0.3),
        "cutoff": (7.0, 3.0),
        "loglogslope": (-6.0, 3.0)
    }

    domain = ift.DomainTuple.make(domain)
    dom_scaling = 1. + np.broadcast_to(pad, (len(domain.axes), ))
    if cf_fluctuations is None:
        cf_fluctuations = cf_fluctuations_sane_default
    if cf_azm is None:
        cf_azm = cf_azm_sane_default

    domain_padded = []
    for d_scl, d in zip(dom_scaling, domain):
        if not isinstance(d, ift.RGSpace) or d.harmonic:
            te = (
                f"unexpected domain encountered in `domain`: {domain}\n"
                "expected a non-harmonic `ift.RGSpace`"
            )
            raise TypeError(te)
        shape_padded = tuple((d_scl * np.array(d.shape)).astype(int))
        domain_padded.append(
            ift.RGSpace(shape_padded, distances=d.distances)
        )
    domain_padded = ift.DomainTuple.make(domain_padded)

    # Set up the signal model
    prefix = "de"  # density estimator
    cfmaker = ift.CorrelatedFieldMaker(prefix)
    for i, d in enumerate(domain_padded):
        if isinstance(cf_fluctuations, (list, tuple)):
            cf_fl = cf_fluctuations[i]
        else:
            cf_fl = cf_fluctuations
        cfmaker.add_fluctuations_matern(d, **cf_fl, prefix=f"ax{i}")
    cfmaker.set_amplitude_total_offset(*cf_azm)
    correlated_field = cfmaker.finalize(0)

    domain_shape = tuple(d.shape for d in domain)
    slc = ift.SliceOperator(correlated_field.target, domain_shape)

    signal = slc @ ift.exp(correlated_field)
    # Cache the result of the correlated field to use it several times
    signal_cache = signal.ducktape_left("signal_cache")
    signal_plchr = ift.FieldAdapter(signal.target, "signal_cache")
    expander = ift.ContractionOperator(slc.target, spaces=None).adjoint
    norm = signal_plchr.integrate().reciprocal()
    signal = (expander @ norm) * signal_plchr
    signal = signal @ signal_cache

    # Honor the difference in measurement time
    if not isinstance(exposure, ift.Operator):
        exposure = ift.ScalingOperator(signal.target, exposure)
    signal_response = exposure @ signal

    model_operators = {
        "signal": signal,
        "correlated_field": correlated_field,
        "exposure": exposure,
        "select_subset": slc,
        "normalization": norm @ signal_cache
    }

    return signal_response, model_operators


if __name__ == "__main__":
    # Preparing the filename string for store results
    filename = "getting_started_density_{}.png"

    # Set up signal domain
    npix1 = 128
    position_space = ift.RGSpace(npix1)

    signal_response, ops = density_estimator(position_space, exposure=10.)
    signal = ops["signal"]
    correlated_field = ops["correlated_field"]
    response = ops["exposure"]
    normalization = ops["normalization"]  # TODO: remove
    # Specify noise
    data_space = response.target

    # Generate mock signal and data
    rng = ift.random.current_rng()
    rng.standard_normal(1000)
    mock_position = ift.from_random(signal_response.domain, 'normal')
    data = ift.Field.from_raw(data_space, rng.poisson(signal_response(mock_position).val))

    plot = ift.Plot()
    plot.add(ift.exp(correlated_field(mock_position)), title='Pre-Slicing Truth')
    plot.add(signal(mock_position), title='Ground Truth')
    plot.add(response.adjoint_times(data), title='Data')
    plot.output(ny=1, nx=3, xsize=10, ysize=10, name=filename.format("setup"))

    # Minimization parameters
    ic_sampling = ift.AbsDeltaEnergyController(name='Sampling',
                                               deltaE=0.01,
                                               iteration_limit=100)
    ic_newton = ift.AbsDeltaEnergyController(name='Newton',
                                             deltaE=0.01,
                                             iteration_limit=35)
    ic_sampling.enable_logging()
    ic_newton.enable_logging()
    minimizer = ift.NewtonCG(ic_newton, enable_logging=True)

    # number of samples used to estimate the KL
    n_samples = 5

    # Set up likelihood and information Hamiltonian
    likelihood = ift.PoissonianEnergy(data) @ signal_response
    ham = ift.StandardHamiltonian(likelihood, ic_sampling)

    # Begin minimization
    initial_mean = ift.MultiField.full(ham.domain, 0.)
    mean = initial_mean

    for i in range(5):
        # Draw new samples and minimize KL
        kl = ift.MetricGaussianKL.make(mean, ham, n_samples, True)
        kl, convergence = minimizer(kl)
        mean = kl.position

        # Plot current reconstruction
        plot = ift.Plot()
        plot.add(ift.exp(correlated_field(mock_position)), title="ground truth")
        plot.add(signal(mock_position), title="ground truth")
        plot.add(signal(kl.position), title="reconstruction")
        plot.add((ic_newton.history, ic_sampling.history,
                  minimizer.inversion_history),
                 label=['kl', 'Sampling', 'Newton inversion'],
                 title='Cumulative energies', s=[None, None, 1],
                 alpha=[None, 0.2, None])
        plot.output(nx=3,
                    ny=2,
                    ysize=10,
                    xsize=15,
                    name=filename.format(f"loop_{i:02d}"))

    # Done, draw posterior samples
    sc = ift.StatCalculator()
    sc_unsliced = ift.StatCalculator()
    for sample in kl.samples:
        sc.add(signal(sample + kl.position))
        sc_unsliced.add(ift.exp(correlated_field(sample + kl.position)))

    # Plotting
    filename_res = filename.format("results")
    plot = ift.Plot()
    plot.add(sc.mean, title="Posterior Mean")
    plot.add(ift.sqrt(sc.var), title="Posterior Standard Deviation")
    plot.add(sc_unsliced.mean, title="Posterior Unsliced Mean")
    plot.add(ift.sqrt(sc_unsliced.var), title="Posterior Unsliced Standard Deviation")

    plot.output(ny=2, nx=2, xsize=15, ysize=15, name=filename_res)
    print("Saved results as '{}'.".format(filename_res))
