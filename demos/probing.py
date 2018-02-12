import nifty4 as ift
import numpy as np


np.random.seed(42)
x = ift.RGSpace((8, 8))

f = ift.Field.from_random(domain=x, random_type='normal')
diagOp = ift.DiagonalOperator(f)

diag = ift.probe_diagonal(diagOp, 1000)
ift.dobj.mprint((f - diag).norm())
