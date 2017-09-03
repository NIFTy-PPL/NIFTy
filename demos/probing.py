# -*- coding: utf-8 -*-

from __future__ import print_function
import nifty2go as ift
import numpy as np

np.random.seed(42)

class DiagonalProber(ift.DiagonalProberMixin, ift.Prober):
    pass


class MultiProber(ift.DiagonalProberMixin, ift.TraceProberMixin, ift.Prober):
    pass


x = ift.RGSpace((8, 8))

f = ift.Field.from_random(domain=x, random_type='normal')
diagOp = ift.DiagonalOperator(domain=x, diagonal=f)

diagProber = DiagonalProber(domain=x)
diagProber(diagOp)
print((f - diagProber.diagonal).norm())

multiProber = MultiProber(domain=x)
multiProber(diagOp)
print((f - multiProber.diagonal).norm())
print(f.sum() - multiProber.trace)
