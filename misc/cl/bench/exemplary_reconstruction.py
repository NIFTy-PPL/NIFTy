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
# Copyright(C) 2024-2025 Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import nifty8 as ift
import numpy as np

if __name__ == "__main__":
    npix = [512, 512]
    ncoils = 16
    dtype, dtype_real = np.complex128, np.float64
    # dtype, dtype_real = np.complex64, np.float32

    device_id = -1
    SNR = 10

    # Signal model
    dom = ift.RGSpace(npix), ift.UnstructuredDomain(ncoils)
    signal = ift.SimpleCorrelatedField(dom[0], 0, (1, 1), (1, 1), (1, 1), (1, 1), (-4, 1))

    # Response
    bc = ift.ContractionOperator(dom, 1).adjoint
    coilsens_fld = ift.from_random(dom, dtype=dtype, device_id=device_id)
    coilsens = ift.makeOp(coilsens_fld)
    fft = ift.FFTOperator(dom, space=0)
    ddom = fft.target
    data_model = fft @ coilsens @ bc @ signal

    # Synthetic data
    dtype_inp = {kk: dtype_real for kk in signal.domain.keys()}
    dtype_inp["xi"] = dtype
    pos_ground_truth = ift.from_random(signal.domain, dtype=dtype_inp, device_id=device_id)
    data = data_model(pos_ground_truth)

    noise_var = data.var(spaces=0) / SNR
    invcov = ift.DiagonalOperator(1/noise_var, domain=data.domain, spaces=0)
    exit()

    e = ift.GaussianEnergy(data=data, inverse_covariance=invcov)
    op = e @ data_model

    n_iterations = 6
    n_samples = lambda iiter: 10 if iiter < 5 else 20

    ic_sampling = ift.AbsDeltaEnergyController(name="Sampling (linear)", deltaE=0.05,
                                               iteration_limit=100)
    ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=0.5, convergence_level=2,
                                             iteration_limit=35)
    ic_sampling_nl = ift.AbsDeltaEnergyController(name='Sampling (nonlin)', deltaE=0.5,
                                                  iteration_limit=15, convergence_level=2)
    minimizer = ift.NewtonCG(ic_newton)
    minimizer_sampling = ift.NewtonCG(ic_sampling_nl)

    samples = ift.optimize_kl(likelihood_energy, n_iterations, n_samples, minimizer, ic_sampling,
                              nonlinear_sampling_minimizer=minimizer_sampling,
                              export_operator_outputs={"signal": signal},
                              output_directory="getting_started_3_results", comm=comm)



    if args.method in ["cupy", "numpy", "numpy-8threads"]:
        res = ift.exec_time(op, domain_dtype=dtype_inp, want_metric=False,
                            device_id=device_id, ntries=ntries)

    elif args.method == "jax":
        import jax
        import jax.numpy as jnp
        import nifty8.re as jft

        if dtype_real == np.float64:
            jax.config.update("jax_enable_x64", True)

        cfm = jft.CorrelatedFieldMaker("")
        cfm.add_fluctuations(npix, (1, 1), (1, 1), (-4, 1), (1, 1), (1, 1))
        cfm.set_amplitude_total_offset(0, (1, 1))
        signal = cfm.finalize()


        class Signal(jft.Model):
            def __init__(self, mod0, c):
                self._mod0 = mod0
                self._c = c
                super().__init__(init=self._mod0.init)

            def __call__(self, x):
                return jnp.fft.fftn(self._c * self._mod0(x)[..., None])


        from nifty8.operators.jax_operator import _anyarray2jax
        lh = jft.Gaussian(_anyarray2jax(data.val), _anyarray2jax(invcov_fld.val)).amend(Signal(signal, coilsens_fld.asnumpy()))
        dom = dict(op.domain)
        dom["spectrum"] = ift.makeDomain(ift.UnstructuredDomain(dom["spectrum"].shape[::-1]))
        dom = ift.makeDomain(dom)
        jop = ift.JaxOperator(dom, op.target, lh)

        res = ift.exec_time(jop, domain_dtype=dtype_inp, want_metric=False, device_id=0, ntries=ntries)

    else:
        raise ValueError

    s = map(str, [args.method, ndof] + list(res.values()))
    s = ",".join(s)
    import os
    if not os.path.isfile("benchmark_data.csv"):
        with open("benchmark_data.csv", "w") as f:
            f.write(",".join(res.keys()) + "\n")
    with open("benchmark_data.csv", "a") as f:
        f.write(s + "\n")
