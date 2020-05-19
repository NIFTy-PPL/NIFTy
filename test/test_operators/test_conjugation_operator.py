import numpy as np
from numpy.testing import assert_allclose

import nifty6 as ift

from ..common import setup_function, teardown_function


def test_conjugation_operator():
    sp = ift.RGSpace(8)
    dom = ift.makeDomain(sp)
    f_real = ift.from_random(dom)
    f_imag = ift.from_random(dom)
    f_complex = f_real + 1.j*f_imag
    op = ift.ScalingOperator(sp,1).conjugate()
    res1 = f_complex.conjugate()
    res2 = op(f_complex)
    assert_allclose(res1.val, res2.val)
