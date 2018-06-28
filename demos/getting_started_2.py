import nifty5 as ift
import numpy as np
# from nifty5.library.nonlinearities import Exponential

#DEFINE THE SIGNAL:

#Define signal space as a regular grid
#s_space = ift.RGSpace(1024)
# s_space = ift.RGSpace([128,128])
s_space = ift.HPSpace(128)
#Define the harmonic space
h_space = s_space.get_default_codomain()

#Prepare Harmonic transformation between the two spaces
HT = ift.HarmonicTransformOperator(h_space, s_space)

#Define domain
domain = ift.MultiDomain.make({'xi': h_space})

#Define positions from a Gaussian distribution
position = ift.from_random('normal', domain)
Nsamples = 5
#Define a power spectrum
def sqrtpspec(k):
    return 10. / (20.+k**2)

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
exposure = 1*np.ones(s_space.shape)
# exposure[int(s_space.shape[0]/3):int(s_space.shape[0]/3+10)] = 10.

#Convert the mask into a field
exposure = ift.Field(s_space,val=exposure)

#Create a diagonal matrix corresponding to the mask
E = ift.DiagonalOperator(exposure)

#Create the response operator and apply the mask on it
R = ift.GeometryRemover(s_space) * E

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


#Define a iteration controller for the minimizer
ic_newton = ift.GradientNormController(name='Newton', tol_abs_gradnorm=1e-3)
minimizer = ift.RelaxedNewton(ic_newton)

#Build the Hamiltonian
H = ift.Hamiltonian(likelihood, ic_cg)
H, convergence = minimizer(H)

#PLOT RESULTS:
   
#Evaluate lambda at final position
lamb_recontructed = lamb.at(H.position).value
sky_reconstructed = sky.at(H.position).value
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

