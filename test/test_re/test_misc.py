import pytest
import nifty8.re as jft
import numpy as np

pmp = pytest.mark.parametrize


@pmp("params", (({"t1":2, 5:"foo"}, ("t1",), {"t1":2}, {5:"foo"}),
                ({}, (), {}, {}),
                ({}, (42,), {},{}),
                ({"t1":2, 5:"foo"}, (), {}, {"t1":2, 5:"foo"}),
                ({"t1":2, 5:"foo"}, ("key",), {}, {"t1":2, 5:"foo"}),
                ({"t1":2, 5:"foo"}, ("t1", 5, False), {"t1":2, 5:"foo"}, {})))
def test_dictsplit(params):
    dct, keys, sel, rest = params
    sel2, rest2 = jft.misc.split(dct, keys)
    if sel2 != sel or rest2 != rest:
        raise RuntimeError("nifty8.re.misc.split failed")

@pmp("params", ((np.zeros(10), True),
                (np.zeros((1,1)), False),
                (np.zeros(()), False),
                ([2,3,4], True),
                (False, False),
                ((), False)))
def test_is1d(params):
    obj, res = params
    if jft.misc.is1d(obj) is not res:
        raise RuntimeError("nifty8.re.misc.is1d failed")
