import nifty as nt
import numpy as np

import unittest

def weighted_np_transform(val, domain, codomain, axes=None):
    if codomain.harmonic:
        # correct for forward fft
        val = domain.weight(val, power=1, axes=axes)
        # Perform the transformation

    Tval = np.fft.fftn(val, axes=axes)

    if not codomain.harmonic:
        # correct for inverse fft
        Tval = codomain.weight(Tval, power=-1, axes=axes)

    return Tval

def test_simple_fft():
    x = nt.RGSpace((16,))
    x_p = nt.FFTOperator.get_default_codomain(x)
    f = nt.Field((x, x), val=1)
    val_p = np.ones((16,16))
    fft = nt.FFTOperator(x)

    assert np.allclose(
        fft(f, spaces=(1,)).val,
        weighted_np_transform(val_p, x, x_p, axes=(1,))
    )
