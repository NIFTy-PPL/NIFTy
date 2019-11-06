import nifty5 as ift
import numpy as np
np.random.seed(42)

sspace = ift.RGSpace((128,))
hspace = sspace.get_default_codomain()
target0 = ift.PowerSpace(hspace)

fa = ift.CorrelatedFieldMaker()
fa.add_fluctuations(target0, 10, 2, 1, 1e-6, 2, 1e-6, -2, 1e-6, 'spatial')
op = fa.finalize(10, 0.1, '')
A = fa.amplitudes[0]

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
