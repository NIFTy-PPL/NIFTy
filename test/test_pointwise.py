import numpy as np
import nifty8 as ift
import pytest
from numpy.testing import assert_allclose

fld_no_complex = ['leakyclip', 'softclip']
lin_no_complex = ['abs', 'sign', 'clip', 'leakyclip', 'softclip', 'unitstep']


def test_asserts():
    dom = ift.UnstructuredDomain((1))
    fld = ift.makeField(dom, np.array([1 + 1j]))
    assert fld.dtype == 'complex128'
    lin = ift.Linearization.make_var(fld)

    for f in fld_no_complex:
        with pytest.raises(TypeError):
            f = (f, -1.0, 1.0) if f[-4:] == 'clip' else (f, )
            _ = fld.ptw(*f)

    for f in lin_no_complex:
        with pytest.raises(TypeError):
            f = (f, -1.0, 1.0) if f[-4:] == 'clip' else (f, )
            _ = lin.ptw(*f)


def test_softclip():
    x = np.linspace(-10., 10., 2000)
    lim = (-5.0, 5.0)
    res = ift.pointwise.softclip(x, *lim)
    assert (res >= lim[0]).all()
    assert (res <= lim[1]).all()

    idx = np.logical_and(x > -0.01, x < 0.01)
    assert_allclose(res[idx], x[idx], rtol=0.02)
