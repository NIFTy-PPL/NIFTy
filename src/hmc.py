from jax import numpy as np
from jax import random, jit, partial, lax
from collections import namedtuple


###
### COMMON FUNCTIONALITY
###

# A datatype for (q, p) = (position, momentum) pairs
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
# @partial(jit, static_argnames=('potential_energy_gradient',))
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
    # TODO: delete step_length
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


###
### SIMPLE HMC
###

# WARNING: requires jaxlib '0.1.66', keyword argument passing doesn't work with alternative static_argnums, which is supported in earlier jax versions
# @partial(jit, static_argnames=('potential_energy', 'potential_energy_gradient'))
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
            + np.sum(qp.momentum**2 / diagonal_momentum_covariance)
        )
    ), momentum


###
### NUTS
###

# A datatype carrying tree metadata
# left, right: endpoints of the trees path
# weight: sum over all exp(-H(q, p)) in the trees path
# proposal_candidate: random sample from the trees path, distributed as exp(-H(q, p))
# turning: TODO: whole tree or also subtrees??
Tree = namedtuple('Tree', ['left', 'right', 'weight', 'proposal_candidate', 'turning'])


def _impl_build_tree_recursive(initial_qp, eps, depth, direction, stepper):
    """Build tree of given depth starting from given initial position.
    
    Parameters
    ----------
    depth: int
        The depth of the tree to be built. Depth is defined as the longest path
        from the root node to any other node. The depth can be expressed as
        log_2(trajectory_length).
        """
    if depth == 0:
        # build a depth 0 tree by leapfrog stepping into the given direction
        left_and_right_qp = stepper(initial_qp, eps, direction)
        # the trajectory contains only a single point, so the left and right endpoints are identical
        return left_and_right_qp, left_and_right_qp, [left_and_right_qp], False
    else:
        # 
        left, right, current_chosen, current_stop = _impl_build_tree_recursive(initial_qp, eps, depth - 1, direction, stepper)
        if direction == -1:
            left, _, new_chosen, new_stop = _impl_build_tree_recursive(left, eps, depth - 1, direction, stepper)
            
        elif direction == 1:
            _, right, new_chosen, new_stop = _impl_build_tree_recursive(right, eps, depth - 1, direction, stepper)
        else:
            raise RuntimeError
        stop = current_stop or new_stop or is_euclidean_uturn(left, right)
        chosen = current_chosen + new_chosen
        return left, right, chosen, stop

def build_tree_recursive(initial_qp, key, eps, maxdepth, stepper):
    left_endpoint, right_endpoint = initial_qp, initial_qp
    stop = False
    chosen = []
    j = 0
    while not stop and j <= maxdepth:
        #print(left_endpoint, right_endpoint)
        key, subkey = random.split(key)
        direction = random.choice(subkey, np.array([-1, 1]))
        print(f"going in direction {int(direction)}")
        if direction == 1:
            _, right_endpoint, new_chosen, new_stop = _impl_build_tree_recursive(right_endpoint, eps, j, direction, stepper)
        elif direction == -1:
            left_endpoint, _, new_chosen, new_stop = _impl_build_tree_recursive(left_endpoint, eps, j, direction, stepper)
        else:
            raise RuntimeError
        if not stop:
            chosen = chosen + new_chosen
        stop = new_stop or is_euclidean_uturn(left_endpoint, right_endpoint)
        j = j + 1
    return left_endpoint, right_endpoint, chosen


def build_tree_iterative(initial_qp, key, eps, maxdepth, stepper):
    left_endpoint, right_endpoint = initial_qp, initial_qp
    stop = False
    chosen = []
    j = 0
    while not stop and j <= maxdepth:
        #print(left_endpoint, right_endpoint)
        key, subkey = random.split(key)
        direction = random.choice(subkey, np.array([-1, 1]))
        print(f"going in direction {int(direction)}")
        if direction == 1:
            other_left_endpoint, other_right_endpoint, other_turning, other_chosen = _impl_build_tree_iterative(right_endpoint, j, eps, direction, stepper)
            if not other_turning:
                right_endpoint = other_right_endpoint
                chosen = chosen + other_chosen
        elif direction == -1:
            other_left_endpoint, other_right_endpoint, other_turning, other_chosen = _impl_build_tree_iterative(left_endpoint, j, eps, direction, stepper)
            if not other_turning:
                left_endpoint = other_left_endpoint
                chosen = chosen + other_chosen
        else:
            raise RuntimeError(f"invalid direction: {direction}")
        print(f"{len(chosen)} chosen states")
        stop = other_turning or is_euclidean_uturn(left_endpoint, right_endpoint)
        j = j + 1
    return left_endpoint, right_endpoint, chosen


# taken from https://arxiv.org/pdf/1912.11554.pdf
def _impl_build_tree_iterative(initial_qp, depth, eps, direction, stepper):
    chosen = []
    z = initial_qp
    S = [initial_qp for _ in range(depth+1)]
    #S = np.empty(shape=(depth,) + initial_qp.shape)
    for n in range(2**depth):
        z = stepper(z, eps, direction)
        chosen.append(z)
        if n % 2 == 0:
            #print(n, bitcount(n))
            S[bitcount(n)] = z
        else:
            # gets the number of candidate nodes
            l = count_trailing_ones(n)
            i_max = bitcount(n-1)
            i_min = i_max - l
            for k in range(i_max, i_min, -1):
                if is_euclidean_uturn(S[k], z):
                    return S[0], z, True, chosen
    return S[0], z, False, chosen


def bitcount(n):
    """Count the number of ones in the binary representation of n.
    
    Examples
    --------
    >>> print(bin(23), bitcount(23))
    0b10111 4
    """
    # TODO: python 3.10 has int.bit_count()
    return bin(n)[2:].count('1')


def count_trailing_ones(n):
    """Count the number of trailing, consecutive ones in the binary representation of n.

    Examples
    --------
    >>> print(bin(23), count_trailing_one_bits(23))
    0b10111 3
    """
    bits_backwards = bin(n)[2:][::-1]
    count = 0
    # now count leading ones
    for b in bits_backwards:
        if b == '1':
            count += 1
        elif b == '0':
            break
        else:
            raise RuntimeError(f"encountered invalid bit \"{b}\" in binary representation of: {n}")
    return count
            

def is_euclidean_uturn(qp_left, qp_right):
    return (
        np.dot(qp_right.momentum, (qp_right.position - qp_left.position)) < 0
        and np.dot(qp_left.momentum, (qp_left.position - qp_right.position)) < 0
    )


def test_run_build_tree_rec():
    from jax import grad
    import matplotlib.pyplot as plt
    dims = (2,)
    potential_energy = lambda q: 0 * 0.5 * np.sum((q - np.ones(shape=dims))**2) + 0.5 * np.sum(((q - np.ones(shape=dims)) / np.array([0.3, 3]))**2)
    kinetic_energy = lambda p: 0.5 * np.sum(p**2)
    potential_energy_gradient = grad(potential_energy)
    stepper = lambda qp, eps, direction: leapfrog_step(potential_energy_gradient, qp, eps*direction)[0]
    key = random.PRNGKey(42)
    key, subkey1, subkey2 = random.split(key, 3)
    initial_qp = QP(position=random.normal(subkey1, dims), momentum=random.normal(subkey2, dims))
    current_qp = initial_qp
    chosen_states = []
    proposed_states = []
    acceptance_probabilities = []
    acceptance_bools = []
    plt.plot(current_qp.position[0], current_qp.position[1], 'rx', label='initial position')
    plt.arrow(current_qp.position[0], current_qp.position[1], 0.1*current_qp.momentum[0], 0.1*current_qp.momentum[1])
    for loop_idx in range(50):
        plt.arrow(current_qp.position[0], current_qp.position[1], 0.1*current_qp.momentum[0], 0.1*current_qp.momentum[1])
        key, subkey = random.split(key)
        left_endpoint, right_endpoint, chosen = build_tree_iterative(current_qp, subkey, 0.01194, 6, stepper)
        key, subkey = random.split(key)
        proposed_qp = chosen[random.choice(subkey, np.array(len(chosen)))]
        proposed_states.append(proposed_qp)
        chosen = np.array(chosen)
        chosen_states.append(chosen)
        acceptance_probability = np.exp(
            potential_energy(current_qp.position) + kinetic_energy(current_qp.momentum)
            - potential_energy(proposed_qp.position) - kinetic_energy(proposed_qp.momentum)
        )
        acceptance_probabilities.append(acceptance_probability)
        print("acceptance probability:", acceptance_probability)
        key, subkey = random.split(key)
        acceptance_threshold = random.uniform(subkey)
        if acceptance_threshold < acceptance_probability:
            current_qp = proposed_qp
            acceptance_bools.append(True)
            print("accepted")
        else:
            print("rejected")
            acceptance_bools.append(False)
        if True or loop_idx == 2:
            plt.plot(current_qp.position[0], current_qp.position[1], 'rx')
            plt.arrow(current_qp.position[0], current_qp.position[1], 0.1*current_qp.momentum[0], 0.1*current_qp.momentum[1], alpha=0.2)
            plt.scatter(chosen[:,0,0], chosen[:,0,1], s=0.2, label='chosen states', alpha=0.9)
        key, subkey = random.split(key)
        current_qp = QP(position=current_qp.position, momentum=random.normal(subkey, shape=dims))
    return initial_qp, chosen_states, proposed_states, acceptance_probabilities, acceptance_bools