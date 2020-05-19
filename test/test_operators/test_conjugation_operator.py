import numpy as np
from numpy.testing import assert_allclose

import nifty6 as ift

from ..common import setup_function, teardown_function


def test_conjugation_operator():
    sp = ift.RGSpace(8)
    dom = ift.makeDomain(sp)
    f = ift.from_random(dom, dtype= np.complex128)
    op = ift.ScalingOperator(sp,1).conjugate()
    res1 = f.conjugate()
    res2 = op(f)
    assert_allclose(res1.val, res2.val)
    ift.extra.consistency_check(op, domain_dtype=np.float64,
                                    target_dtype=np.float64)
    ift.extra.consistency_check(op, domain_dtype=np.complex128,
                                target_dtype=np.complex128, only_r_linear=True)
