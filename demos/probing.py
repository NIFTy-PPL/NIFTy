import nifty2go as ift
import numpy as np

np.random.seed(42)


class DiagonalProber(ift.DiagonalProberMixin, ift.Prober):
    pass


class MultiProber(ift.DiagonalProberMixin, ift.TraceProberMixin, ift.Prober):
    pass


x = ift.RGSpace((8, 8))

f = ift.Field.from_random(domain=x, random_type='normal')
diagOp = ift.DiagonalOperator(f)

diagProber = DiagonalProber(domain=x)
diagProber(diagOp)
ift.dobj.mprint((f - diagProber.diagonal).norm())

multiProber = MultiProber(domain=x)
multiProber(diagOp)
ift.dobj.mprint((f - multiProber.diagonal).norm())
ift.dobj.mprint(f.sum() - multiProber.trace)
