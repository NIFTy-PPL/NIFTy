from jax import numpy as np
import jax.tree_util as tree_util
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


def total_energy_of_qp(qp, potential_energy, kinetic_energy):
    qparr = np.array(qp)
    return potential_energy(qparr[0,:]) + kinetic_energy(qparr[1,:])

def build_tree_iterative(initial_qp, key, eps, maxdepth, stepper, potential_energy, kinetic_energy):
    current_tree = Tree(left=initial_qp, right=initial_qp, weight=np.exp(-total_energy_of_qp(initial_qp, potential_energy, kinetic_energy)), proposal_candidate=initial_qp, turning=False)
    stop = False
    j = 0
    while not stop and j <= maxdepth:
        #print(left_endpoint, right_endpoint)
        key, subkey = random.split(key)
        # random.bernoulli is fine, this is just for rng consistency across commits TODO: use random.bernoulli
        go_right = random.choice(subkey, np.array([False, True]))
        print(f"going in direction {1 if go_right else -1}")
        # new_tree = current_tree extended (i.e. doubled) in direction
        new_tree = extend_tree_iterative(key, current_tree, j, eps, go_right, stepper, potential_energy, kinetic_energy)
        stop = new_tree.turning or is_euclidean_uturn(new_tree.left, new_tree.right)
        j = j + 1
        # TODO: I think this needs to be conditional on stop somehow but not sure
        if not stop:
            current_tree = new_tree
    return current_tree


# taken from https://arxiv.org/pdf/1912.11554.pdf
def extend_tree_iterative(key, initial_tree, depth, eps, go_right, stepper, potential_energy, kinetic_energy):
    # 1. choose start point of integration
    if go_right:
        initial_qp = initial_tree.right
    else:
        initial_qp = initial_tree.left
    # 2. build / collect chosen states
    chosen = []
    z = initial_qp
    S = [initial_qp for _ in range(depth+1)]
    #S = np.empty(shape=(depth,) + initial_qp.shape)
    for n in range(2**depth):
        z = stepper(z, eps, 1 if go_right else -1)
        chosen.append(z)
        if n % 2 == 0:
            #print(n, bitcount(n))
            S[bitcount(n)] = z
        else:
            # gets the number of candidate nodes
            l = count_trailing_ones(n)
            i_max_incl = bitcount(n-1)
            i_min_incl = i_max_incl - l + 1
            S_to_check_against = S[i_min_incl:i_max_incl + 1][::-1]
            assert len(S_to_check_against) == l
            contains_uturn = any(tree_util.tree_map(
                lambda s: is_euclidean_uturn(s, z),
                S_to_check_against,
                is_leaf=lambda qp: isinstance(qp, QP)
            ))
            if contains_uturn:
                # TODO: what to return here? old tree or new tree but with turning set?
                return initial_tree
    key, subkey = random.split(key)
    new_subtree = make_tree_from_list(subkey, chosen, go_right, potential_energy, kinetic_energy, False)
    return merge_trees(key, initial_tree, new_subtree, go_right, False)

def make_tree_from_list(key, qp_list, go_right, potential_energy, kinetic_energy, turning_hint):
    # WARNING: only to be called from extend_tree_iterative, with turning_hint logic correct
    # 3. random.choice with probability weights to get sample
    qp_arr = np.array(qp_list)
    #new_subtree_energies = lax.map(potential_energy, chosen_array[:,0,:]) + lax.map(kinetic_energy, chosen_array[:,1,:])
    new_subtree_energies = lax.map(partial(total_energy_of_qp, potential_energy=potential_energy, kinetic_energy=kinetic_energy), qp_arr)
    print(f"new_subtree_energies.shape: {new_subtree_energies.shape}")
    # proportianal to the joint probabilities:
    new_subtree_weights = np.exp(-new_subtree_energies)
    print(qp_arr.shape)
    # unfortunately choice only works with 1d arrays so we need to use indexing TODO: factor out?
    random_idx = random.choice(key, qp_arr.shape[0], p=new_subtree_weights)
    new_subtree_sample = qp_list[random_idx]
    print(f"chose sample n° {random_idx}")
    # 4. calculate total weight
    new_subtree_total_weight = np.sum(new_subtree_weights)
    left, right = lax.cond(
        pred = go_right,
        true_fun = lambda _qp_list: (_qp_list[0], _qp_list[-1]),
        false_fun = lambda _qp_list: (_qp_list[-1], _qp_list[0]),
        operand = qp_list
    )
    return Tree(left=left, right=right, weight=new_subtree_total_weight, proposal_candidate=new_subtree_sample, turning=turning_hint)

def merge_trees(key, current_subtree, new_subtree, go_right, turning_hint):
    """Merges two trees, propagating the proposal_candidate"""
    # WARNING: only to be called from extend_tree_iterative, with turning_hint logic correct
    # 5. decide which sample to take based on total weights (merge trees)
    key, subkey = random.split(key)
    print(f"prob of choosing new sample: {new_subtree.weight / (new_subtree.weight + current_subtree.weight)}")
    new_sample = lax.cond(
        pred = random.bernoulli(subkey, new_subtree.weight / (new_subtree.weight + current_subtree.weight)),
        # choose the new sample
        true_fun = lambda current_and_new_tup: current_and_new_tup[1],
        # choose the old sample
        false_fun = lambda current_and_new_tup: current_and_new_tup[0],
        operand = (current_subtree.proposal_candidate, new_subtree.proposal_candidate)
    )
    # 6. define new tree
    left, right = lax.cond(
        pred = go_right,
        true_fun = lambda op: (op['current_subtree'].left, op['new_subtree'].right),
        false_fun = lambda op: (op['new_subtree'].left, op['current_subtree'].right),
        operand = {'current_subtree': current_subtree, 'new_subtree': new_subtree}
    )
    merged_tree = Tree(left=left, right=right, weight=new_subtree.weight + current_subtree.weight, proposal_candidate=new_sample, turning=turning_hint)
    return merged_tree


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
        (np.dot(qp_right.momentum, (qp_right.position - qp_left.position)) < 0.)
        & (np.dot(qp_left.momentum, (qp_left.position - qp_right.position)) < 0.)
    )
