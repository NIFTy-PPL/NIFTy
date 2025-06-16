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
# Copyright(C) 2025 LambdaFields GmbH
#
# Author: Philipp Arras

import nifty8.cl as ift
import numpy as np
import pytest
from numpy.testing import assert_allclose

from .common import setup_function, teardown_function

pmp = pytest.mark.parametrize


@pmp("ddtype", (complex, float))
def test_minisanity_gaussian(ddtype):
    ndtype = float

    dom = ift.UnstructuredDomain(10)
    mask = np.ones(dom.shape)
    mask[:2] = 0
    mask = ift.makeField(dom, mask)

    d = ift.from_random(dom, dtype=ddtype).val_rw()
    N = ift.from_random(dom, dtype=ndtype).exp().val_rw()
    d[2] = np.nan
    N[2] = np.nan
    d = ift.makeField(dom, d)
    N = ift.makeField(dom, N)
    sig = N.sqrt()
    N = ift.makeOp(N, sampling_dtype=ddtype)

    N.draw_sample()
    N.inverse.draw_sample()

    e = ift.GaussianEnergy(d*mask, N) @ ift.makeOp(mask)

    sl = ift.SampleList([ift.from_random(e.domain, dtype=ddtype)
                         for _ in range(3)])
    out, ms = ift.extra.minisanity(e, sl, return_values=True)

    igndof = ms["nigndof"]
    assert igndof["data_residuals"]["<None>"] == 3
    assert igndof["latent_variables"]["<None>"] == 0

    ndof = ms["ndof"]
    assert ndof["data_residuals"]["<None>"] == 7
    assert ndof["latent_variables"]["<None>"] == 10

    val = ms["scmean"]
    dres = ift.StatCalculator()
    var = ift.StatCalculator()
    redchisq_d = ift.StatCalculator()
    redchisq_var = ift.StatCalculator()
    for ss in sl.iterator():
        effdof = 7
        nres = (sig*(ss-d)*mask).val
        dres_tmp = np.nansum(nres) / effdof
        dres.add(dres_tmp)

        redchisq_d_tmp = np.nansum(nres.conj() * nres) / effdof
        redchisq_d.add(redchisq_d_tmp)

        effdof = 10
        var_tmp = np.sum(ss.val) / effdof
        var.add(var_tmp)

        redchisq_var_tmp = np.nansum(ss.val.conj() * ss.val) / effdof
        redchisq_var.add(redchisq_var_tmp)

    # NOTE: dres.mean is the "average (over samples) averaged (over data
    # space) normalized data residual"
    assert_allclose(val["data_residuals"]["<None>"]["mean"], dres.mean)
    assert_allclose(val["data_residuals"]["<None>"]["std"], np.sqrt(dres.var))
    assert_allclose(val["latent_variables"]["<None>"]["mean"], var.mean)
    assert_allclose(val["latent_variables"]["<None>"]["std"], np.sqrt(var.var))

    val = ms["redchisq"]
    assert_allclose(val["data_residuals"]["<None>"]["mean"], redchisq_d.mean)
    assert_allclose(val["data_residuals"]["<None>"]["std"], np.sqrt(redchisq_d.var))
    assert_allclose(val["latent_variables"]["<None>"]["mean"], redchisq_var.mean)
    assert_allclose(val["latent_variables"]["<None>"]["std"], np.sqrt(redchisq_var.var))
