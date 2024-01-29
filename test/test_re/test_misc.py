import pytest
import nifty8.re as jft
import numpy as np

pmp = pytest.mark.parametrize


@pmp(
    "params",
    (
        ({"t1": 2, 5: "foo"}, ("t1",), {"t1": 2}, {5: "foo"}),
        ({}, (), {}, {}),
        ({}, (42,), {}, {}),
        ({"t1": 2, 5: "foo"}, (), {}, {"t1": 2, 5: "foo"}),
        ({"t1": 2, 5: "foo"}, ("key",), {}, {"t1": 2, 5: "foo"}),
        ({"t1": 2, 5: "foo"}, ("t1", 5, False), {"t1": 2, 5: "foo"}, {}),
    ),
)
def test_dictsplit(params):
    dct, keys, sel, rest = params
    sel2, rest2 = jft.misc.split(dct, keys)
    assert sel2 == sel and rest2 == rest


@pmp(
    "params",
    (
        (np.zeros(10), True),
        (np.zeros((1, 1)), False),
        (np.zeros(()), False),
        ([2, 3, 4], True),
        (False, False),
        ((), True),
    ),
)
def test_is1d(params):
    obj, res = params
    assert jft.misc.is_iterable_of_non_iterables(obj) is res
