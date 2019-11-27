import nifty6 as ift
import numpy as np
np.random.seed(42)

sspace = ift.RGSpace((128,))

fa = ift.CorrelatedFieldMaker.make(10, 0.1, '')
fa.add_fluctuations(sspace, 10, 2, 1, 1e-6, 2, 1e-6, -2, 1e-6, 'spatial')
op = fa.finalize()
A = fa.amplitude

cstpos = ift.from_random('normal', op.domain)
p1, p2 = [ift.Plot() for _ in range(2)]
lst1 = []
skys1, skys2 = [], []
for _ in range(8):
    pos = ift.from_random('normal', op.domain)

    foo = ift.MultiField.union([cstpos, pos.extract(A.domain)])
    skys2.append(op(foo))

    sky = op(pos)
    skys1.append(sky)
    lst1.append(A.force(pos))

for pp, ll in [(p1, skys1), (p2, skys2)]:
    mi, ma = None, None
    if False:
        mi, ma = np.inf, -np.inf
        for ss in ll:
            mi = min([mi, np.amin(ss.val)])
            ma = max([ma, np.amax(ss.val)])
    for ss in ll:
        pp.add(ss, zmin=mi, zmax=ma)

p1.add(lst1)
p2.add(lst1)
p1.output(name='full.png')
p2.output(name='xi_fixed.png')
