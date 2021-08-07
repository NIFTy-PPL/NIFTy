# %%
import jax.numpy as np
from jax import random
from jax import jit
from jifty1.hmc import leapfrog_step_pytree, generate_nuts_sample, QP

# %%
def test_run_build_tree_rec():
    from jax import grad
    import matplotlib.pyplot as plt
    dims = (2,)
    to_arr = lambda q_or_p: np.array([q_or_p['x'], q_or_p['y']])
    to_dict = lambda q_or_p: {'x': q_or_p[0], 'y': q_or_p[1]}
    potential_energy_arr = lambda q: 0 * 0.5 * np.sum((q - np.ones(shape=dims))**2) + 0.5 * np.sum(((q - np.ones(shape=dims)) / np.array([0.3, 3]))**2)
    potential_energy = lambda q: potential_energy_arr(to_arr(q))
    kinetic_energy_arr = lambda p: 0.5 * np.sum(p**2)
    kinetic_energy = lambda p: kinetic_energy_arr(to_arr(p))
    potential_energy_gradient = grad(potential_energy)
    stepper = lambda qp, eps, direction: leapfrog_step_pytree(potential_energy_gradient, qp, eps*direction)[0]
    key = random.PRNGKey(42)
    key, subkey1, subkey2 = random.split(key, 3)
    initial_qp = QP(position=to_dict(random.normal(subkey1, dims)), momentum=to_dict(random.normal(subkey2, dims)))
    current_qp = initial_qp
    proposed_states = []
    acceptance_probabilities = []
    acceptance_bools = []
    plt.plot(to_arr(current_qp.position)[0], to_arr(current_qp.position)[1], 'rx', label='initial position')
    plt.arrow(to_arr(current_qp.position)[0], to_arr(current_qp.position)[1], 0.1*to_arr(current_qp.momentum)[0], 0.1*to_arr(current_qp.momentum)[1])
    generate_nuts_sample_jitted = jit(generate_nuts_sample, static_argnames=['maxdepth', 'stepper', 'potential_energy', 'kinetic_energy'])
    for loop_idx in range(10):
        plt.arrow(to_arr(current_qp.position)[0], to_arr(current_qp.position)[1], 0.1*to_arr(current_qp.momentum)[0], 0.1*to_arr(current_qp.momentum)[1])
        key, subkey = random.split(key)
        tree = generate_nuts_sample_jitted(current_qp, subkey, 0.01194, 6, stepper, potential_energy, kinetic_energy)
        key, subkey = random.split(key)
        proposed_states.append(tree.proposal_candidate)
        acceptance_probability = np.exp(
            potential_energy(current_qp.position) + kinetic_energy(current_qp.momentum)
            - potential_energy(tree.proposal_candidate.position) - kinetic_energy(tree.proposal_candidate.momentum)
        )
        acceptance_probabilities.append(acceptance_probability)
        print("acceptance probability:", acceptance_probability)
        key, subkey = random.split(key)
        acceptance_threshold = random.uniform(subkey)
        if acceptance_threshold < acceptance_probability:
            current_qp = tree.proposal_candidate
            acceptance_bools.append(True)
            print("accepted")
        else:
            print("rejected")
            acceptance_bools.append(False)
        if True or loop_idx == 2:
            plt.plot(to_arr(current_qp.position)[0], to_arr(current_qp.position)[1], 'rx')
            plt.arrow(to_arr(current_qp.position)[0], to_arr(current_qp.position)[1], 0.1*to_arr(current_qp.momentum)[0], 0.1*to_arr(current_qp.momentum)[1], alpha=0.2)
        key, subkey = random.split(key)
        # resample momentum
        current_qp = QP(position=current_qp.position, momentum=to_dict(random.normal(subkey, shape=dims)))
    return initial_qp, proposed_states, acceptance_probabilities, acceptance_bools

# %%
initial_qp, proposed_states, acceptance_probabilities, acceptance_bools = test_run_build_tree_rec()
# %%
