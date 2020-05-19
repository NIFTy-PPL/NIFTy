import numpy as np
from numpy.testing import assert_allclose

import nifty6 as ift

from ..common import setup_function, teardown_function

def test_contraction_operator():
    x1 = ift.RGSpace((9,), distances=2.)
    x2 = ift.RGSpace((2, 12), distances=(0.3,))
    dom1 = ift.makeDomain(x1)
    dom2 = ift.makeDomain((x1, x2))
    f1 = ift.from_random(dom1)
    f2 = ift.from_random(dom2)
    op1 = ift.ScalingOperator(dom1, 1).sum()
    op2 = ift.ScalingOperator(dom2, 1).sum()
    op3 = ift.ScalingOperator(dom2, 1).sum(spaces=1)
    res1 = f1.sum()
    res2 = op1(f1)
    assert_allclose(res1.val, res2.val)
    res3 = f2.sum()
    res4 = op2(f2)
    assert_allclose(res3.val, res4.val)
    res5 = f2.sum(spaces=1)
    res6 = op3(f2)
    assert_allclose(res5.val, res6.val)
    for op in [op1, op2, op3]:
        ift.extra.consistency_check(op, domain_dtype=np.float64,
                                    target_dtype=np.float64)
        ift.extra.consistency_check(op, domain_dtype=np.complex128,
                                    target_dtype=np.complex128)
