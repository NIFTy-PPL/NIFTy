import numpy as np

from ..field import Field

__all__ = ['test_adjointness', 'test_inverse']


def test_adjointness(op, domain_dtype=np.float64, target_dtype=np.float64, atol=0, rtol=1e-7):
    f1 = Field.from_random("normal", domain=op.domain, dtype=domain_dtype)
    f2 = Field.from_random("normal", domain=op.target, dtype=target_dtype)
    res1 = f1.vdot(op.adjoint_times(f2))
    res2 = op.times(f1).vdot(f2)
    np.testing.assert_allclose(res1, res2, atol=atol, rtol=rtol)

    # Return relative error
    return (res1 - res2) / (res1 + res2) * 2


def test_inverse(op, dtype_domain=np.float64, dtype_target=np.float64, atol=0, rtol=1e-7):
    foo = Field.from_random(domain=op.target, random_type='normal', dtype=dtype_target)
    res = op(op.inverse_times(foo)).val
    ones = Field.ones(op.domain).val
    np.testing.assert_allclose(res, ones, atol=atol, rtol=rtol)

    foo = Field.from_random(domain=op.domain, random_type='normal', dtype=dtype_domain)
    res = op.inverse_times(op(foo)).val
    ones = Field.ones(op.target).val
    np.testing.assert_allclose(res, ones, atol=atol, rtol=rtol)

    # Return relative error
    return (res - ones) / (res + ones) * 2
