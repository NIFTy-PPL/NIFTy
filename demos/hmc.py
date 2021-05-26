#%%
from jax.config import config
config.update("jax_enable_x64", True)
import sys
from jax import numpy as np
from jax import random, jit, partial, lax
import jax
import matplotlib.pyplot as plt
import jifty1 as jft
from functools import partial
from collections import namedtuple
from jax.tree_util import register_pytree_node
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10, 7)


#%%
dims = (512,)
#datadims = (4,)
loglogslope = 2.
power_spectrum = lambda k: 1. / (k**loglogslope + 1.)
modes = np.arange((dims[0] / 2) + 1., dtype=float)
harmonic_power = power_spectrum(modes)
harmonic_power = np.concatenate((harmonic_power, harmonic_power[-2:0:-1]))


#%%
correlated_field = lambda x: jft.correlated_field.hartley(
    # x is a signal in fourier space
    # each modes amplitude gets multiplied by it's harmonic_power
    # and the whole signal is transformed back
    harmonic_power * x
)

# %% [markdown]
# signal_response = lambda x: np.exp(1. + correlated_field(x))
signal_response = lambda x: (1. + correlated_field(x))
# The signal response is $ \vec{d} = \begin{pmatrix} 1 \\ 1 \\ 1 \\ 1 \end{pmatrix} \cdot s + \vec{n} $ where $s \in \mathbb{R}$ and $\vec{n} \sim \mathcal{G}(0, N)$
# signal_response = lambda x: np.ones(shape=datadims) * x
# ???
noise_cov_inv_sqrt = lambda x: 10**-1 * x

#%%
# create synthetic data
seed = 43
key = random.PRNGKey(seed)
key, subkey = random.split(key)
# normal random fourier amplitude
pos_truth = random.normal(shape=dims, key=subkey)
signal_response_truth = signal_response(pos_truth)
key, subkey = random.split(key)
# 1. / noise_cov_inv_sqrt(np.ones(dims)) becomes the standard deviation of the noise gaussian
noise_truth = 1. / noise_cov_inv_sqrt(np.ones(dims)) * random.normal(shape=dims, key=subkey)
data = signal_response_truth + noise_truth

#%%
plt.plot(signal_response_truth, label='signal response')
#plt.plot(noise_truth, label='noise', linewidth=0.5)
plt.plot(data, label='noisy data', linewidth=1)
plt.xlabel('real space domain')
plt.ylabel('field value')
plt.legend()
plt.title("signal and data")
plt.show()

#%%
def Gaussian(data, noise_cov_inv_sqrt):
    # Simple but not very generic Gaussian energy
    # primals
    def hamiltonian(primals):
        p_res = primals - data
        # TODO: is this the weighting with noies amplitude thing again?
        l_res = noise_cov_inv_sqrt(p_res)
        return 0.5 * np.sum(l_res**2)

    return jft.Likelihood(
        hamiltonian,
    )

# negative log likelihood
nll = Gaussian(data, noise_cov_inv_sqrt) @ signal_response

#%%
ham = jft.StandardHamiltonian(likelihood=nll).jit()
ham = nll
ham_gradient = jax.grad(ham)

#%%
###
### HMC implementation
###

# TODO: Notes
# * no reason to ever take the derivative of gaussian kinetric energy (can be expressed analytically)

# TODO: mass matrix, contra / covariant tansformations
# potential_energy = ham
# kinetic energy with given mass matrix
kinetic_energy_of_p_m = lambda p, m: np.sum(p**2 * 1 / m)

#%%

QP = namedtuple('QP', ['position', 'momentum'])

def flip_momentum(qp: QP) -> QP:
    return QP(position=qp.position, momentum=-qp.momentum)

# TODO: depend on current qs, sample from some kind of kinetic energy object
# TODO: don't depend on current qs, make this a oneliner (i.e. delete this function) using random.normal or something
def sample_momentum_from_diagonal(*, key, diagonal_momentum_covariance):
    """
    Draw a momentum sample from the kinetic energy of the hamiltonian.

    Parameters
    ----------
    key: ndarray
        a PRNGKey used as the random key.
    diagonal_momentum_covariance: ndarray
        the momentum covariance (also: mass matrix) to use for sampling.
        Diagonal matrix stored as an ndarray vector containing the entries of the diagonal.
    """
    unit_normal_vector = random.normal(key, diagonal_momentum_covariance.shape)
    return np.sqrt(1 / diagonal_momentum_covariance) * unit_normal_vector


# TODO: pass gradient instead of calculating gradient in function
# TODO: how to randomize step size (neal sect. 3.2)
# WARNING: requires jaxlib '0.1.66', keyword argument passing doesn't work with alternative static_argnums, which is supported in earlier jax versions
@partial(jit, static_argnames=('potential_energy_gradient',))
def leapfrog_step(
        potential_energy_gradient,
        qp: QP,
        step_length,
    ):
    """
    Perform one iteration of the leapfrog integrator forwards in time.

    Parameters
    ----------
    momentum: ndarray
        Point in momentum space from which to start integration.
        Same shape as `position`.
    position: ndarray
        Point in position space from which to start integration.
        Same shape as `momentum`.
    potential_energy: Callable[[ndarray], float]
        Potential energy part of the hamiltonian (V). Depends on position only.
    step_length: float
        Step length (usually called epsilon) of the leapfrog integrator.
    """
    position = qp.position
    momentum = qp.momentum

    momentum_halfstep = (
        momentum
        - (step_length / 2.) * potential_energy_gradient(position)  # type: ignore
    )
    #print("momentum_halfstep:", momentum_halfstep)

    position_fullstep = position + step_length * momentum_halfstep  # type: ignore
    #print("position_fullstep:", position_fullstep)

    momentum_fullstep = (
        momentum_halfstep
        - (step_length / 2.) * potential_energy_gradient(position_fullstep)  # type: ignore
    )
    #print("momentum_fullstep:", momentum_fullstep)

    qp_fullstep = QP(position=position_fullstep, momentum=momentum_fullstep)
    # return the last two paramters unchanged for iteration in lax.fori_loop
    # TODO: apply last two args partially?
    #return potential_energy, momentum_fullstep, position_fullstep, step_length
    return qp_fullstep, step_length


def accept_or_deny(*,
        key,
        old_qp: QP,
        proposed_qp: QP,
        total_energy
    ):
    """Perform acceptance step.
    
    Returning the new or the old (p, q) pairs depending on wether the new ones
    were accepted or not.

    Parameters
    ----------
    old_momentum: ndarray,
    old_position: ndarray,
    proposed_momentum: ndarray,
    proposed_position: ndarray,
    total_energy: Callable[[qp], float]
        The sum of kinetic and potential energy as a function of position and
        momentum.
    """
    # TODO: new energy quickly becomes NaN, can be fixed by keeping step size small (?)
    # how to handle this case?
    #print(f"old_e {total_energy(old_qp):3.4e}")
    #print(f"new_e {total_energy(proposed_qp):3.4e}")
    acceptance_threshold = np.min(np.array(
        [
            1,
            np.exp(
                total_energy(old_qp)
                - total_energy(proposed_qp)
            )
        ]
    ))

    acceptance_level = random.uniform(key)

    #print(f"level: {acceptance_level:3.4e}, thresh: {acceptance_threshold:3.4e}")

    # TODO: define namedtuple with rejected and accepted and 
    return ((old_qp, proposed_qp), acceptance_level < acceptance_threshold)


# WARNING: requires jaxlib '0.1.66', keyword argument passing doesn't work with alternative static_argnums, which is supported in earlier jax versions
@partial(jit, static_argnames=('potential_energy', 'potential_energy_gradient'))
def generate_next_sample(*,
        key,
        position,
        potential_energy,
        potential_energy_gradient,
        diagonal_momentum_covariance,
        number_of_integration_steps,
        step_length
    ):
    """
    Generate a sample given the initial position.

    Parameters
    ----------
    key: ndarray
        a PRNGKey used as the random key
    position: ndarray
        The the starting position of this step of the markov chain.
    potential_energy: Callable[[ndarray], float]
        The potential energy, which is the distribution to be sampled from.
    diagonal_momentum_covariance: ndarray
        The mass matrix used in the kinetic energy
    number_of_integration_steps: int
        The number of steps the leapfrog integrator should perform.
    step_length: float
        The step size (usually epsilon) for the leapfrog integrator.
    """
    key, subkey = random.split(key)
    momentum = sample_momentum_from_diagonal(
        key = subkey,
        diagonal_momentum_covariance = diagonal_momentum_covariance
    )
    qp = QP(position=position, momentum=momentum)
    
    loop_body = partial(leapfrog_step, potential_energy_gradient)
    idx_ignoring_loop_body = lambda idx, args: loop_body(*args)

    # todo: write in python (maybe?)
    new_qp, _step_length = lax.fori_loop(
        lower = 0,
        upper = number_of_integration_steps,
        body_fun = idx_ignoring_loop_body,
        init_val = (
            qp,
            step_length,
        )
    )

    # this flipping is needed to make the proposal distribution symmetric
    # doesn't have any effect on acceptance though because kinetic energy depends on momentum^2
    # might have an effect with other kinetic energies though
    proposed_qp = flip_momentum(new_qp)

    return accept_or_deny(
        key = key,
        old_qp = qp,
        proposed_qp = proposed_qp,
        total_energy = lambda qp: (
            potential_energy(qp.position)
            + kinetic_energy_of_p_m(qp.momentum, diagonal_momentum_covariance)
        )
    ), momentum


#%%
key, subkey = random.split(key)
initial_position = random.uniform(key=subkey, shape=pos_truth.shape)

#%%
position_samples = [initial_position]
momentum_samples = [None]
unintegrated_momenta = [None]
# this is a slight misnomer as it not only contains the rejected (i.e.
# integrated samples) but also the integration starting values (unintegrated
# samples) in the case that the proposed values were rejected.
rejected_position_samples = [None]
rejected_momentum_samples = [None]
accepted = [True]

#%%
for _ in range(10000):
    key, subkey = random.split(key)
    (qp_old_and_proposed_sample, was_accepted), unintegrated_momentum = generate_next_sample(
        key = subkey,
        position = position_samples[-1],
        potential_energy = ham,
        potential_energy_gradient = ham_gradient,
        diagonal_momentum_covariance = np.ones(shape = initial_position.shape) * 10,
        number_of_integration_steps = 187,
        step_length = 0.0001723
    )
    position_samples.append(qp_old_and_proposed_sample[was_accepted.item()].position)
    momentum_samples.append(qp_old_and_proposed_sample[was_accepted.item()].momentum)
    rejected_position_samples.append(qp_old_and_proposed_sample[not was_accepted].position)
    rejected_momentum_samples.append(qp_old_and_proposed_sample[not was_accepted].momentum)
    unintegrated_momenta.append(unintegrated_momentum)
    accepted.append(was_accepted)
# first value is only needed during the loop
position_samples = np.array(position_samples[1:])
momentum_samples = np.array(momentum_samples[1:])
rejected_position_samples = np.array(rejected_position_samples[1:])
rejected_momentum_samples = np.array(rejected_momentum_samples[1:])
accepted = np.array(accepted[1:])
unintegrated_momenta = np.array(unintegrated_momenta[1:])

print(f"acceptance ratio: {np.sum(accepted)/len(accepted)}")

# %% [markdown]
# # Position samples
signal_response_of_samples = lax.map(signal_response, position_samples)
mean_of_signal_response = np.mean(signal_response_of_samples, axis=0)
std_dev_of_signal_response = np.std(signal_response_of_samples, axis=0)
plt.plot(signal_response(pos_truth), label="truth")
#plt.plot(data, label="data")
#for idx, s in enumerate(position_samples[::200]):
    #plt.plot(signal_response(s), label=str(idx))
plt.plot(mean_of_signal_response, label='sample mean of signal response')
plt.fill_between(np.arange(len(mean_of_signal_response)), y1=mean_of_signal_response - std_dev_of_signal_response, y2=mean_of_signal_response + std_dev_of_signal_response, color='grey', alpha=0.5)
plt.title('position samples')
plt.xlabel('position')
plt.ylabel('signal response')
plt.legend(loc='upper right')

# %% [markdown]
# # 1D position and momentum time series
if position_samples[0].shape != (1,):
    raise ValueError("time series only availible for 1d data")
plt.plot(position_samples, label='position')
#plt.plot(momentum_samples, label='momentum', linewidth=0.2)
#plt.plot(unintegrated_momenta, label='unintegrated momentum', linewidth=0.2)
plt.title('position and momentum time series')
plt.xlabel('time')
plt.ylabel('position, momentum')
plt.legend()

# %% [markdown]
# # energy time series
potential_energies = lax.map(ham, position_samples)
kinetic_energies = np.sum(momentum_samples**2, axis=1)
rejected_potential_energies = lax.map(ham, rejected_position_samples)
rejected_kinetic_energies = np.sum(rejected_momentum_samples**2, axis=1)
plt.plot(potential_energies , label='pot')
plt.plot(kinetic_energies , label='kin', linewidth=1)
plt.plot(kinetic_energies + potential_energies, label='total', linewidth=1)
plt.plot(rejected_potential_energies , label='rejected_pot')
plt.plot(rejected_kinetic_energies , label='rejected_kin', linewidth=2)
plt.plot(rejected_kinetic_energies + rejected_potential_energies, label='rejected_total', linewidth=0.2)
plt.xlabel('time')
plt.ylabel('energy')
plt.yscale('log')
plt.legend()

#%%
def check_leapfrog_energy_conservation():
    np.set_printoptions(precision=2)
    _dims = (2,)
    spring_constants = 100. * np.ones(shape=_dims)
    #potential_energy = lambda q: -1 / np.linalg.norm(q)
    potential_energy = lambda q: np.sum(q.T @ np.linalg.inv(np.array([[1, 0.95], [0.95, 1]])) @ q / 2.)
    potential_energy_gradient = jax.grad(potential_energy)
    mass_matrix = np.ones(shape=_dims)
    kinetic_energy = lambda p: np.sum(p**2 / mass_matrix / 2.)
    position = np.array([-1.5, -1.55])
    momentum = np.array([-1, 1]) #random.normal(key=key, shape=_dims) / mass_matrix
    positions = [position]
    momenta = [momentum]
    for _ in range(25):
        new_qp, _ = leapfrog_step(
            potential_energy_gradient=potential_energy_gradient,
            qp = QP(position=positions[-1], momentum=momenta[-1]),
            step_length=0.25
        )
        positions.append(new_qp.position)
        momenta.append(new_qp.momentum)
    positions = np.array(positions)
    momenta = np.array(momenta)
    old_pot_energy = potential_energy(position) 
    new_pot_energy = potential_energy(new_qp.position)
    g = jax.grad(potential_energy)
    print("intial gradient:", g(position))
    old_kin_energy = np.sum(momentum**2 / mass_matrix)
    new_kin_energy = np.sum(new_qp.momentum**2 / mass_matrix)
    #print("old pos:", position)
    #print("new pos:", new_qp.position)
    #print("old mom:", momentum)
    #print("new mom:", new_qp.momentum)
    print("old pot E: ", old_pot_energy)
    print("old kin E: ", old_kin_energy)
    print("old total:", old_kin_energy + old_pot_energy)
    print("new pot E: ", new_pot_energy)
    print("new kin E: ", new_kin_energy)
    print("new total:", new_kin_energy + new_pot_energy)
    print(positions)
    print(momenta)
    kinetic_energies = np.sum(momenta**2, axis=1)
    potential_energies = -1 / np.linalg.norm(positions, axis=1)
    return momenta, positions, kinetic_energies, potential_energies

momenta, positions, kinetic_energies, potential_energies = check_leapfrog_energy_conservation()


# %% [markdown]
# # Position Coordinates
plt.plot(positions[:,0], positions[:,1])
plt.xlabel("position[:,0]")
plt.ylabel("position[:,1]")


# %% [markdown]
# # Momentum coordinates
plt.plot(momenta[:,0], momenta[:,1])
plt.xlabel("momenta[:,0]")
plt.ylabel("momenta[:,1]")


# %% [markdown]
# # Value of Hamiltonian
# ## does not look exactly the same as in Neal (2011) unfortunately!
plt.plot(kinetic_energies, label='kin')
plt.plot(potential_energies, label='pot')
plt.plot(kinetic_energies + potential_energies, label='total')
plt.xlabel('time')
plt.ylabel('energy')
plt.legend()
# %%
