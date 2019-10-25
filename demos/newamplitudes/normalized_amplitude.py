import numpy as np

import nifty5 as ift

if __name__ == '__main__':
    np.random.seed(42)
    ndim = 1
    sspace = ift.RGSpace(
        np.linspace(16, 20, num=ndim).astype(np.int),
        np.linspace(2.3, 7.99, num=ndim))
    hspace = sspace.get_default_codomain()
    target = ift.PowerSpace(hspace)
    op = ift.NormalizedAmplitude(target, 16, 1, 1, -3, 1, 0, 1, 0, 1)
    if not isinstance(op.target[0], ift.PowerSpace):
        raise RuntimeError
    fld = ift.from_random('normal', op.domain)
    ift.extra.check_jacobian_consistency(op, fld)

    pd = ift.PowerDistributor(hspace, power_space=target)
    ht = ift.HartleyOperator(pd.target)
    if np.testing.assert_allclose((pd @ op)(fld).integrate(), 1):
        raise RuntimeError
    if np.testing.assert_allclose(
        (ht @ pd @ op)(fld).to_global_data()[ndim*(0,)], 1):
        raise RuntimeError
