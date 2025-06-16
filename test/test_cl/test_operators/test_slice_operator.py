#!/usr/bin/env python3

import pytest
from numpy.testing import assert_array_equal

import nifty8.cl as ift

pmp = pytest.mark.parametrize


@pmp("dim,new_shape", (((32,), ((16,),)), ((32, 32), ((16, 16),))))
def test_slice_operator(dim, new_shape):
    domain = ift.RGSpace(dim)
    print(new_shape)
    slice_op = ift.SliceOperator(domain, new_shape, center=False, preserve_dist=True)

    field = ift.Field.from_random(domain=domain, random_type="normal", std=1, mean=0)
    sliced_field = slice_op(field)
    assert sliced_field.asnumpy().shape == new_shape[0]
    assert_array_equal(
        field.asnumpy()[tuple(slice(0, ns) for ns in new_shape[0])], sliced_field.asnumpy()
    )
