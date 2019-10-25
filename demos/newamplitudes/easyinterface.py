import nifty5 as ift

sspace = ift.RGSpace((128, 128), (0.2, 0.2))
hspace = sspace.get_default_codomain()
target = ift.PowerSpace(hspace)

vol = hspace.scalar_dvol**-0.5
ft = ift.HartleyOperator(hspace, sspace).scale(vol)

A = ift.NormalizedAmplitude(target, 16, 1, 1, -3, 1, 0, 1, 0, 1)

avgA = ift.full(A.target, 0.)
n = 1000
for _ in range(n):
    avgA = avgA + A(ift.from_random('normal', A.domain))
avgA = avgA/n

corfldfixA = ift.CorrelatedField(sspace, avgA)
corfld = ift.CorrelatedField(sspace, A)

p = ift.Plot()
p1 = ift.Plot()
lst, lst1 = [avgA**2], []
for _ in range(8):
    pos = ift.from_random('normal', corfld.domain)

    skyfixA = corfldfixA.force(pos)
    p.add(skyfixA)
    lst.append(ift.power_analyze(ft.inverse(skyfixA)))

    sky = corfld(pos)
    p1.add(sky)
    lst1.append(A.force(pos))
p.add(lst)
p1.add(lst1)
p.output(name='mean_power_spectrum.png', xsize=15, ysize=15)
p1.output(name='full.png', xsize=15, ysize=15)
