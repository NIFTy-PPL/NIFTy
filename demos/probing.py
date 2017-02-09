# -*- coding: utf-8 -*-

from nifty import Field, RGSpace, DiagonalProberMixin, TraceProberMixin,\
                  Prober, DiagonalOperator


class DiagonalProber(DiagonalProberMixin, Prober):
    pass


class MultiProber(DiagonalProberMixin, TraceProberMixin, Prober):
    pass


x = RGSpace((8, 8))

f = Field.from_random(domain=x, random_type='normal')
diagOp = DiagonalOperator(domain=x, diagonal=f)

diagProber = DiagonalProber(domain=x)
diagProber(diagOp)
print (f - diagProber.diagonal).norm()

multiProber = MultiProber(domain=x)
multiProber(diagOp)
print (f - multiProber.diagonal).norm()
print f.sum() - multiProber.trace



