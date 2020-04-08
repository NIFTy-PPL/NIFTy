from numpy.testing import assert_, assert_allclose
import numpy as np
from copy import deepcopy
import nifty6 as ift


class CountingOp(ift.Operator):
    #FIXME: Not a LinearOperator since ChainOps not supported yet
    def __init__(self, domain):
        self._domain = self._target = ift.sugar.makeDomain(domain)
        self._count = 0

    def apply(self, x):
        self._count += 1
        return x

    @property
    def count(self):
        return self._count


def test_operator_tree_optimiser():
    dom = ift.RGSpace(10, harmonic=True)
    hdom = dom.get_default_codomain()
    cop1 = CountingOp(dom)
    op1 = (ift.UniformOperator(dom, -1, 2)@cop1).ducktape('a')
    cop2 = CountingOp(dom)
    op2 = ift.FieldZeroPadder(dom, (11,))@cop2
    cop3 = CountingOp(op2.target)
    op3 = ift.ScalingOperator(op2.target, 3)@cop3
    cop4 = CountingOp(op2.target)
    op4 = ift.ScalingOperator(op2.target, 1.5) @ cop4
    op1 = op1 * op1
    # test layering in between two levels
    op = op3@op2@op1 + op2@op1 + op3@op2@op1 + op2@op1
    op = op + op
    op = op4@(op4@op + op4@op)
    fld = ift.from_random('normal', op.domain, np.float64)
    op_orig = deepcopy(op)
    op = ift.operator_tree_optimiser._optimise_operator(op)
    assert_allclose(op(fld).val, op_orig(fld).val, rtol=np.finfo(np.float64).eps)
    assert_(1 == ( (cop4.count-1) * cop3.count * cop2.count * cop1.count))
