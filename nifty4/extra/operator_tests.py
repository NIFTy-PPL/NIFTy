import numpy as np

from ..field import Field

__all__ = ['adjoint_implementation', 'inverse_implemenation', 'full_implementation']


def adjoint_implementation(op, domain_dtype=np.float64, target_dtype=np.float64, atol=0, rtol=1e-7):
    f1 = Field.from_random("normal", domain=op.domain, dtype=domain_dtype)
    f2 = Field.from_random("normal", domain=op.target, dtype=target_dtype)
    res1 = f1.vdot(op.adjoint_times(f2))
    res2 = op.times(f1).vdot(f2)
    np.testing.assert_allclose(res1, res2, atol=atol, rtol=rtol)

    # Return relative error
    return (res1 - res2) / (res1 + res2) * 2


def inverse_implementation(op, domain_dtype=np.float64, target_dtype=np.float64, atol=0, rtol=1e-7):
    foo = Field.from_random(domain=op.target, random_type='normal', dtype=target_dtype)
    res = op(op.inverse_times(foo)).val
    np.testing.assert_allclose(res, foo.val, atol=atol, rtol=rtol)

    foo = Field.from_random(domain=op.domain, random_type='normal', dtype=domain_dtype)
    res = op.inverse_times(op(foo)).val
    np.testing.assert_allclose(res, foo.val, atol=atol, rtol=rtol)

    # Return relative error
    return (res - foo.val) / (res + foo.val) * 2


def full_implementation(op, domain_dtype=np.float64, target_dtype=np.float64, atol=0, rtol=1e-7):
    res1 = inverse_implementation(op, domain_dtype, target_dtype, atol, rtol)
    res2 = adjoint_implementation(op, domain_dtype, target_dtype, atol, rtol)
    res3 = adjoint_implementation(op.inverse, target_dtype, domain_dtype, atol, rtol)

    return res1, res2, res3
