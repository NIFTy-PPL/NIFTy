import numpy as np

from ..field import Field

__all__ = ['test_adjointness', 'test_inverse']


def test_adjointness(self, domain_dtype=np.float64, target_dtype=np.float64, atol=0, rtol=1e-7):
    f1 = Field.from_random("normal", domain=self.domain, dtype=domain_dtype)
    f2 = Field.from_random("normal", domain=self.target, dtype=target_dtype)
    res1 = f1.vdot(self.adjoint_times(f2))
    res2 = self.times(f1).vdot(f2)
    np.testing.assert_allclose(res1, res2, atol=atol, rtol=rtol)

    # Return relative error
    return (res1 - res2) / (res1 + res2) * 2


def test_inverse(self, dtype_domain=np.float64, dtype_target=np.float64, atol=0, rtol=1e-7):
    foo = Field.from_random(domain=self.target, random_type='normal', dtype=dtype_target)
    res = self.times(self.inverse_times(foo)).val
    ones = Field.ones(self.domain).val
    np.testing.assert_allclose(res, ones, atol=atol, rtol=rtol)

    foo = Field.from_random(domain=self.domain, random_type='normal', dtype=dtype_domain)
    res = self.inverse_times(self.times(foo)).val
    ones = Field.ones(self.target).val
    np.testing.assert_allclose(res, ones, atol=atol, rtol=rtol)

    # Return relative error
    return (res - ones) / (res + ones) * 2
