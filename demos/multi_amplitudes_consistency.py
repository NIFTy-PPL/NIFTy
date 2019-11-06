import nifty5 as ift
import numpy as np
#np.random.seed(42)

offset_std = 40
intergated_fluct_std0 = 10.
intergated_fluct_std1 = 2.

#sspace = ift.RGSpace((32,64),(1.1,0.3))
sspace = ift.HPSpace(64)
sspace = ift.GLSpace(64)
hspace = sspace.get_default_codomain()
target0 = ift.PowerSpace(hspace)

fsspace = ift.RGSpace((12,),(0.4,))
fhspace = fsspace.get_default_codomain()
target1 = ift.PowerSpace(fhspace)

fa = ift.CorrelatedFieldMaker()
fa.add_fluctuations(target0, intergated_fluct_std0, 1E-8,
                    1.1, 2., 2.1, .5, -2, 1., 'spatial')
fa.add_fluctuations(target1, intergated_fluct_std1, 1E-8,
                    3.1, 1., .5, .1, -4, 1., 'freq')
op = fa.finalize(offset_std, 1E-8, '')

flucts = [intergated_fluct_std0, intergated_fluct_std1]
tot_flm,totflsig = fa.effective_total_fluctuation(flucts,[1E-8,1E-8])

space = op.target
totaltoalvol = 1.
for s in space[:]:
    totaltoalvol *= s.total_volume

nsam = 1000

zm_std_mean = 0.
fluct_space = 0.
fluct_freq = 0.
fluct_total = 0.

for i in range(nsam):
    x = ift.from_random('normal',op.domain)
    res = op(x)

    zm = res.integrate()/totaltoalvol
    zm2 = res.mean()
    
    fl = ((res-zm)**2).integrate() / totaltoalvol
    
    zm_std_mean += zm**2
    fluct_total += fl

    r = res.integrate(1)/fsspace.total_volume
    r0 = r.integrate()/sspace.total_volume
    tm = ((r-r0)**2).integrate() / sspace.total_volume
    fluct_space += tm
    
    fr = res.integrate(0)/sspace.total_volume
    fr0 = fr.integrate()/fsspace.total_volume
    ftm = ((fr-fr0)**2).integrate() / fsspace.total_volume
    fluct_freq += ftm
    

fluct_total = np.sqrt(fluct_total/nsam)
fluct_space = np.sqrt(fluct_space/nsam)
fluct_freq = np.sqrt(fluct_freq/nsam)
zm_std_mean = np.sqrt(zm_std_mean/nsam)

print("Expected  offset Std: "+str(offset_std))
print("Estimated offset Std: "+str(zm_std_mean))

print("Expected  integrated fluct. space Std: "+str(intergated_fluct_std0))
print("Estimated integrated fluct. space Std: "+str(fluct_space))

print("Expected  integrated fluct. frequency Std: "+str(intergated_fluct_std1))
print("Estimated integrated fluct. frequency Std: "+str(fluct_freq))

print("Expected  total fluct. Std: "+str(tot_flm))
print("Estimated total fluct. Std: "+str(fluct_total))