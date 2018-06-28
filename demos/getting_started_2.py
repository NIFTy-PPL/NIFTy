import nifty5 as ift
import sys
import numpy as np
import global_newton as gn
from nifty5.library.nonlinearities import Exponential

#DEFINE THE SIGNAL:

#Define signal space as a regular grid
#s_space = ift.RGSpace(10024)
s_space = ift.RGSpace([128,128])

#Define the harmonic space
h_space = s_space.get_default_codomain()

#Prepare Harmonic transformation between the two spaces
HT = ift.HarmonicTransformOperator(h_space, s_space)

#Define domain
domain = ift.MultiDomain.make({'xi': h_space})

#Define positions from a Gaussian distribution
position = ift.from_random('normal', domain)

#Define a power spectrum
def sqrtpspec(k):
    return 16. / (20.+k**2)

#Define a power space
p_space = ift.PowerSpace(h_space)

#Define the power distribution between the harmonic and the power spaces
pd = ift.PowerDistributor(h_space, p_space)

#Create a field with the defined power spectrum
a = ift.PS_field(p_space, sqrtpspec)

#Define the amplitudes
A = pd(a)

#Unpack the positions xi from the Multifield
xi = ift.Variable(position)['xi']

#Multiply the positions by the amplitudes in the harmonic domain
logsky_h = xi * A

#Transform to the real domain
logsky = HT(logsky_h)

#Create a sky model by applying the exponential (Poisson)
sky = ift.PointwiseExponential(logsky)

#DEFINE THE RESPONSE OPERATOR:

#Define a mask to cover a patch of the real space
mask = np.ones(s_space.shape)
mask[int(s_space.shape[0]/3):int(s_space.shape[0]/3+10)] = 0.

#Convert the mask into a field
mask = ift.Field(s_space,val=mask)

#Create a diagonal matrix corresponding to the mask
M = ift.DiagonalOperator(mask)

#Create the response operator and apply the mask on it
R = ift.ScalingOperator(1., s_space) * M

#CREATE THE MOCK DATA:

#Define the data space
d_space = R.target[0]

#Apply the response operator to the signal
#lamb corresponds to the mean in the Poisson distribution
lamb = R(sky)

#Draw coordinates of the mock data from a Gaussian distribution
mock_position = ift.from_random('normal', lamb.position.domain)

#Generate mock data from a Poisson distribution using lamb as a mean
data = np.random.poisson(lamb.at(mock_position).value.val.astype(np.float64))

#Store the data as a field
data = ift.Field.from_local_data(d_space, data)

#RECONSTRUCT THE SIGNAL:

#Define the positions where we perform the analysis from a Gaussian distribution
position = ift.from_random('normal', lamb.position.domain)

#Define the Poisson likelihood knowing the mean and the data
likelihood = ift.library.PoissonLogLikelihood(lamb, data)

#Define a iteration controller with a maximum number of iterations
ic_cg = ift.GradientNormController(iteration_limit=50)

#Define a iteration controller with convergence criteria
ic_samps = ift.GradientNormController(iteration_limit=500, tol_abs_gradnorm=1e-4)

#Define a iteration controller for the minimizer
ic_newton = ift.GradientNormController(name='Newton', tol_abs_gradnorm=1e-3)
minimizer = ift.RelaxedNewton(ic_newton)

#Build the Hamiltonian
H = gn.Hamiltonian(likelihood, ic_cg, ic_samps)

#Iterate the reconstruction
for _ in range(N):
    #Draw samples from the curvature
    samples = [H.curvature.draw_sample(from_inverse=True)
               for _ in range(Nsamples)]
                          
    sc_samplesky = ift.StatCalculator()
    for s in samples:
        sc_samplesky.add(sky.at(s+position).value)
    ift.plot(sc_samplesky.mean, name='sample_mean.png')

    #Compute the Kullbachh Leibler divergence
    KL = gn.SampledKullbachLeiblerDivergence(H, samples, ic_cg)
    
    #Update the position
    position = KL.position
    
    #Obtain the value of the KL and the convergence at the new position
    KL, convergence = minimizer(KL)

#PLOT RESULTS:
   
#Evaluate lambda at final position
lamb_recontructed = lamb.at(KL.position).value

#Evaluate lambda at the data position for comparison
lamb_mock = lamb.at(mock_position).value

#Plot the data, reconstruction and underlying signal
ift.plot([data, lamb_mock, lamb_recontructed], name="poisson.png",
         label=['Data', 'Mock signal', 'Reconstruction'],
         alpha=[.5, 1, 1])
         
#Plot power spectrum for posterior test         
a_mock = a
a_recon = a
ift.plot([a_mock**2, a_recon**2, ift.power_analyze(logsky_h.at(KL.position).value)],
         name='power_spectrum.png', label=['Mock', 'Reconstruction', 'power_analyze'])

