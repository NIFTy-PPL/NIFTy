import matplotlib.pyplot as plt
import jax
from jax import numpy as np

from jifty1 import hmc

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
        new_qp = hmc.leapfrog_step(
            qp = hmc.QP(position=positions[-1], momentum=momenta[-1]),
            potential_energy_gradient=potential_energy_gradient,
            step_length=0.25,
            mass_matrix=mass_matrix
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
plt.show()


# %% [markdown]
# # Momentum coordinates
plt.plot(momenta[:,0], momenta[:,1])
plt.xlabel("momenta[:,0]")
plt.ylabel("momenta[:,1]")
plt.show()


# %% [markdown]
# # Value of Hamiltonian
# ## does not look exactly the same as in Neal (2011) unfortunately!
plt.plot(kinetic_energies, label='kin')
plt.plot(potential_energies, label='pot')
plt.plot(kinetic_energies + potential_energies, label='total')
plt.xlabel('time')
plt.ylabel('energy')
plt.legend()
plt.show()