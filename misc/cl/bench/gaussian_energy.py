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

import nifty.cvl as ift
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('npix', type=int)
    parser.add_argument('method', type=str)
    args = parser.parse_args()

    print("npix", args.npix, "method", args.method)
    npix = [args.npix, args.npix]
    ndof = np.prod(npix)

    ncoils = 16
    dtype, dtype_real = np.complex128, np.float64
    # dtype, dtype_real = np.complex64, np.float32

    if args.method in ["numpy", "numpy-8threads"]:
        device_id = -1
    elif args.method == "cupy":
        device_id = 0
    elif args.method == "jax":
        device_id = 0
    else:
        raise ValueError

    if args.method == "numpy-8threads":
        ift.set_nthreads(8)

    if "numpy" in args.method and args.npix > 500:
        ntries = 3
    else:
        ntries = 20

    dom = ift.RGSpace(npix), ift.UnstructuredDomain(ncoils)
    bc = ift.ContractionOperator(dom, 1).adjoint
    coilsens_fld = ift.from_random(dom, dtype=dtype, device_id=device_id)
    coilsens = ift.makeOp(coilsens_fld)
    fft = ift.FFTOperator(dom, space=0)
    ddom = fft.target
    data = ift.from_random(ddom, dtype=dtype, device_id=device_id)
    invcov_fld = ift.from_random(ddom, dtype=dtype_real, device_id=device_id).exp()
    invcov = ift.makeOp(invcov_fld)
    e = ift.GaussianEnergy(data=data, inverse_covariance=invcov)
    signal = ift.SimpleCorrelatedField(dom[0], 0, (1, 1), (1, 1), (1, 1), (1, 1), (-4, 1))
    op = e @ fft @ coilsens @ bc @ signal
    dtype_inp = {kk: dtype_real for kk in signal.domain.keys()}
    dtype_inp["xi"] = dtype

    if args.method in ["cupy", "numpy", "numpy-8threads"]:
        res = ift.exec_time(op, domain_dtype=dtype_inp, want_metric=False,
                            device_id=device_id, ntries=ntries)

    elif args.method == "jax":
        import jax
        import nifty.re as jft
        import jax.numpy as jnp

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


        from nifty.cl.operators.jax_operator import _anyarray2jax
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
