import nifty5 as ift
import numpy as np
np.random.seed(42)

sspace = ift.RGSpace((128, 128), (0.2, 0.2))
hspace = sspace.get_default_codomain()
target = ift.PowerSpace(hspace)

A = ift.NormalizedAmplitude(target, 16, 1, 1, -3, 1, 0, 1, 0, 1)
A = ift.WPAmplitude(target, [0, -2, 0], [1E-5, 1, 1], 1, 0.99,
                    ['rest', 'smooth', 'wienersigma'])

avgA = ift.full(A.target, 0.)
n = 1000
for _ in range(n):
    avgA = avgA + A(ift.from_random('normal', A.domain))
avgA = avgA/n
corfldfixA = ift.CorrelatedField(sspace, avgA)
corfld = ift.CorrelatedField(sspace, A)

corfld = ift.CorrelatedFieldNormAmplitude(sspace, A, 0, 1)
corfldfixA = ift.CorrelatedFieldNormAmplitude(sspace, avgA, 0, 1)

cstpos = ift.from_random('normal', corfld.domain)
p, p1, p2 = [ift.Plot() for _ in range(3)]
lst, lst1 = [avgA**2], []
skys, skys1, skys2 = [], [], []
for _ in range(8):
    pos = ift.from_random('normal', corfld.domain)

    skyfixA = corfldfixA.force(pos)
    skys.append(skyfixA)
    ft = ift.HartleyOperator(hspace, sspace).scale(hspace.scalar_dvol**-0.5)
    lst.append(ift.power_analyze(ft.inverse(skyfixA)))

    foo = ift.MultiField.union([cstpos, pos.extract(A.domain)])
    skys2.append(corfld(foo))

    sky = corfld(pos)
    skys1.append(sky)
    lst1.append(A.force(pos))

for pp, ll in [(p, skys), (p1, skys1), (p2, skys2)]:
    mi, ma = None, None
    if True:
        mi, ma = np.inf, -np.inf
        for ss in ll:
            mi = min([mi, np.amin(ss.val)])
            ma = max([ma, np.amax(ss.val)])
    for ss in ll:
        pp.add(ss, zmin=mi, zmax=ma)

p.add(lst)
p1.add(lst1)
p2.add(lst1)
p1.output(name='full.png')
p.output(name='A_fixed.png')
p2.output(name='xi_fixed.png')
