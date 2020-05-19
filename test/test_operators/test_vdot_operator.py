import numpy as np
from numpy.testing import assert_allclose

import nifty6 as ift

from ..common import setup_function, teardown_function

def test_vdot_operator():
    dom = ift.makeDomain(ift.RGSpace(8))
    fa_1 = ift.FieldAdapter(dom, 'f1')
    fa_2 = ift.FieldAdapter(dom, 'f2')
    op1 = fa_1.vdot(fa_2)
    f = ift.from_random(op1.domain, dtype=np.float64)
    res1 = f['f1'].vdot(f['f2'])
    res2 = op1(f)
    assert_allclose(res1.val, res2.val)
    ift.extra.check_jacobian_consistency(op1, f)
    #another Test for linearization?
