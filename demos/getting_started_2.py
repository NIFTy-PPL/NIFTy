import nifty5 as ift
import sys
import numpy as np
import global_newton as gn
from nifty5.library.nonlinearities import Exponential

np.random.seed(42)

N = 2
Nsamples = 5

s_space = ift.RGSpace(10024)
h_space = s_space.get_default_codomain()

domain = ift.MultiDomain.make({'xi': h_space})
position = ift.from_random('normal', domain)
HT = ift.HarmonicTransformOperator(h_space, s_space)

def sqrtpspec(k):
    return 16. / (20.+k**2)


# Define amplitude model
p_space = ift.PowerSpace(h_space)
pd = ift.PowerDistributor(h_space, p_space)
a = ift.PS_field(p_space, sqrtpspec)
A = pd(a)

# Define sky model
xi = ift.Variable(position)['xi']
logsky_h = xi * A
logsky = HT(logsky_h)
nonlin = Exponential()
sky = ift.PointwiseExponential(logsky)
R = ift.ScalingOperator(1., s_space)
d_space = R.target[0]
lamb = R(sky)

# Generate mock data
MOCK_POSITION = ift.from_random('normal', lamb.position.domain)
data = np.random.poisson(lamb.at(MOCK_POSITION).value.val.astype(np.float64))
data = ift.Field.from_local_data(d_space, data)

# Define Hamiltonian
position = ift.from_random('normal', lamb.position.domain)
likelihood = ift.library.PoissonLogLikelihood(lamb, data)

ic_cg = ift.GradientNormController(iteration_limit=50)
ic_samps = ift.GradientNormController(iteration_limit=500, tol_abs_gradnorm=1e-4)

ic_newton = ift.GradientNormController(name='Newton', tol_abs_gradnorm=1e-3)
minimizer = ift.RelaxedNewton(ic_newton)

H = gn.Hamiltonian(likelihood, ic_cg, ic_samps)

for _ in range(N):
    samples = [H.curvature.draw_sample(from_inverse=True)
               for _ in range(Nsamples)]
    sc_samplesky = ift.StatCalculator()
    for s in samples:
        sc_samplesky.add(sky.at(s+position).value)
    ift.plot(sc_samplesky.mean, name='sample_mean.png')

    KL = gn.SampledKullbachLeiblerDivergence(H, samples, ic_cg)
    KL, convergence = minimizer(KL)
    position = KL.position

# Plot results
E = KL
l1 = lamb.at(E.position).value
l2 = lamb.at(MOCK_POSITION).value
ift.plot([data, l2, l1], name="poisson.png",
         label=['Data', 'Mock signal', 'Reconstruction'],
         alpha=[.5, 1, 1])
if power_spectrum_estimation:
    a_mock = a.at(MOCK_POSITION).value
    a_recon = a.at(E.position).value
else:
    a_mock = a
    a_recon = a
ift.plot([a_mock**2, a_recon**2, ift.power_analyze(logsky_h.at(E.position).value)],
         name='power_spectrum.png', label=['Mock', 'Reconstruction', 'power_analyze'])

