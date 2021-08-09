from jax import numpy as np
import jax.tree_util as tree_util
from jax import lax, random, jit, partial, flatten_util, grad
from .disable_jax_control_flow import cond, while_loop, fori_loop
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


def leapfrog_step_pytree(
    potential_energy_gradient,
    qp: QP,
    step_length
    ):
    position = qp.position
    momentum = qp.momentum

    momentum_halfstep = tree_util.tree_map(
        lambda mom, potgrad: mom - (step_length / 2.) * potgrad,
        momentum,
        potential_energy_gradient(position)
    )

    position_fullstep = tree_util.tree_map(
        lambda pos, mom_halfstep: pos + step_length * mom_halfstep,  # type: ignore
        position,
        momentum_halfstep
    )

    momentum_fullstep = tree_util.tree_map(
        lambda mom_halfstep, potgrad: mom_halfstep - (step_length / 2.) * potgrad,
        momentum_halfstep,
        potential_energy_gradient(position_fullstep)
    )

    qp_fullstep = QP(position=position_fullstep, momentum=momentum_fullstep)
    return qp_fullstep, step_length


def unzip_qp_pytree(tree_of_qp):
    """Turn a tree containing QP pairs into a QP pair of trees"""
    return QP(
        position = tree_util.tree_map(lambda qp: qp.position, tree_of_qp, is_leaf=lambda qp: isinstance(qp, QP)),
        momentum = tree_util.tree_map(lambda qp: qp.momentum, tree_of_qp, is_leaf=lambda qp: isinstance(qp, QP))
    )


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
def generate_hmc_sample(*,
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
    new_qp, _step_length = fori_loop(
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
# turning currently means either the left, right endpoint are a uturn or any subtree is a uturn, see TODO above
# TODO: rename proposal_candidate, taking into account that NUTS is not Metropolis-Hastings.
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
    return potential_energy(qp.position) + kinetic_energy(qp.momentum)


def generate_nuts_sample(initial_qp, key, eps, maxdepth, stepper, potential_energy, kinetic_energy):
    """
    Warning
    -------
    Momentum must be resampled from conditional distribution BEFORE passing into this function!
    This is different from `generate_hmc_sample`!

    Generate a sample given the initial position.

    An implementation of the No-Uturn-Sampler

    Parameters
    ----------
    initial_qp: QP
        starting (position, momentum) pair
        WARNING: momentum must be resampled from conditional distribution BEFORE passing into this function!
    key: ndarray
        a PRNGKey used as the random key
    eps: float
        The step size (usually called epsilon) for the leapfrog integrator.
    maxdepth: int
        The maximum depth of the trajectory tree before expansion is terminated
        and value is sampled even if the U-turn condition is not met.
        The maximum number of points (/integration steps) per trajectory is
            N = 2**maxdepth
        Memory requirements of this function are linear in maxdepth, i.e. logarithmic in trajectory length.
        JIT: static argument
    stepper: Callable[[QP, float, int(1 / -1)] QP]
        The function that performs (Leapfrog) steps. Takes as arguments (in order)
            starting point: QP
            step size: float
            direction: int (but only 1 or -1!)
        JIT: static argument
    potential_energy: Callable[[pytree], float]
        The potential energy, of the distribution to be sampled from.
        Takes only the position part (QP.position) as argument
    kinetic_energy: Callable[[pytree], float]
        The kinetic energy, of the distribution to be sampled from.
        Takes only the momentum part (QP.momentum) as argument
    
    Returns
    -------
    current_tree: Tree
        The final tree, carrying a sample from the target distribution.

    See Also
    --------
    No-U-Turn Sampler original paper (2011): https://arxiv.org/abs/1111.4246
    NumPyro Iterative NUTS paper: https://arxiv.org/abs/1912.11554
    Combination of samples from two trees, Sampling from trajectories according to target distribution in this paper's Appendix: https://arxiv.org/abs/1701.02434
    """
    # initialize depth 0 tree, containing 2**0 = 1 points
    current_tree = Tree(left=initial_qp, right=initial_qp, weight=np.exp(-total_energy_of_qp(initial_qp, potential_energy, kinetic_energy)), proposal_candidate=initial_qp, turning=False)

    # loop stopping condition
    stop = False
    # loop tree depth, increases each iteration
    j = 0

    loop_state = (key, current_tree, j, stop)

    def _cond_fn(loop_state):
        _key, _current_tree, j, stop = loop_state
        # while (not stop) and j <= maxdepth
        return (~stop) & np.less_equal(j, maxdepth)

    def _body_fun(loop_state):
        key, current_tree, j, stop = loop_state
        key, subkey = random.split(key)
        # random.bernoulli is fine, this is just for rng consistency across commits TODO: use random.bernoulli
        go_right = random.choice(subkey, np.array([False, True]))

        # build tree of depth j, adjacent to current_tree
        new_subtree = iterative_build_tree(key, current_tree, j, eps, go_right, stepper, potential_energy, kinetic_energy, maxdepth)

        # combine current_tree and new_subtree into a depth j+1 tree only if new_subtree has no turning subtrees (including itself)
        current_tree = cond(
            pred = new_subtree.turning,
            true_fun = lambda old_and_new: old_and_new[0],
            # TODO: turning_hint
            false_fun = lambda old_and_new: merge_trees(key, old_and_new[0], old_and_new[1], go_right, False),
            operand = (current_tree, new_subtree),
        )
        # stop if new subtree was turning -> we sample from the old one and don't expand further
        # stop if new total tree is turning -> we sample from the combined trajectory and don't expand further
        # TODO: move call to is_euclidean_uturn_pytree into merge_trees from above, remove the turning hint and just check current_tree.turning here
        stop = new_subtree.turning | is_euclidean_uturn_pytree(current_tree.left, current_tree.right)
        j = j + 1
        return (key, current_tree, j, stop)

    _key, current_tree, _j, _stop = while_loop(_cond_fn, _body_fun, loop_state)
    return current_tree


def index_into_pytree_time_series(idx, ptree):
    return tree_util.tree_map(lambda arr: arr[idx], ptree)


# Essentially algorithm 2 from https://arxiv.org/pdf/1912.11554.pdf
def iterative_build_tree(key, initial_tree, depth, eps, go_right, stepper, potential_energy, kinetic_energy, maxdepth):
    """
    Starting from either the left or right endpoint of a given tree, builds a new adjacent tree of the same size.

    Parameters
    ----------
    initial_tree: Tree
        Tree to be extended (doubled) on the left or right.
    depth:
        Depth of the new tree to be built.
    eps: float
        The step size (usually called epsilon) for the leapfrog integrator.
    go_right: bool
        If go_right start at the right end, going right else start at the left end, going left.
    stepper: Callable[[QP, float, int(1 / -1)] QP]
        The function that performs (Leapfrog) steps. Takes as arguments (in order)
            starting point: QP
            step size: float
            direction: int (but only 1 or -1!)
    potential_energy: Callable[[pytree], float]
        The potential energy, of the distribution to be sampled from.
        Takes only the position part (QP.position) as argument
    kinetic_energy: Callable[[pytree], float]
        The kinetic energy, of the distribution to be sampled from.
        Takes only the momentum part (QP.momentum) as argument
    maxdepth: int
        An upper bound on the 'depth' argument, but has no effect on the functions behaviour.
        It's only required to statically set the size of the `S` array (actually pytree of arrays).
    """
    # 1. choose start point of integration
    initial_qp = cond(
        pred = go_right,
        true_fun = lambda left_and_right: left_and_right[1],
        false_fun = lambda left_and_right: left_and_right[0],
        operand = (initial_tree.left, initial_tree.right)
    )
    z = initial_qp
    # 2. build / collect chosen states
    # TODO: rename chosen to a more sensible name such as new_tree ...
    # TODO: WARNING: this will be overwritten in the first iteration of the loop, the assignment to chosen is only temporary and we're using z since it's the only QP that's availible right now. This would also be solved by moving the first iteration outside of the loop.
    chosen = Tree(z,z,0.,z,turning=False)
    # Storage for left endpoints of subtrees. Size is determined statically by the `maxdepth` parameter.
    S = tree_util.tree_map(lambda initial_q_or_p_leaf: np.empty((maxdepth + 1,) + initial_q_or_p_leaf.shape), unzip_qp_pytree(initial_qp))

    def _loop_body(state):
        n, _turning, chosen, z, S, key = state

        z = stepper(z, eps, np.where(go_right, x=1, y=-1))

        # TODO: maybe just move the first iteration outside of the loop?
        chosen = cond(
            pred = n == 0,
            true_fun = lambda c_and_z: Tree(left=z, right=z, weight=np.exp(-total_energy_of_qp(z, potential_energy, kinetic_energy)), proposal_candidate=z, turning=False),
            false_fun = lambda c_and_z: chosen,
            operand = (chosen, z)
        )

        key, subkey = random.split(key)
        # TODO: this is okay on the first iteration (n==0) becaues chosen.proposal_candidate == z. But it is also unnecessary.
        chosen = add_single_qp_to_tree(subkey, chosen, z, go_right, potential_energy, kinetic_energy)

        def _even_fun(n_and_S):
            n, S = n_and_S
            # TODO: does z have to be passed in or is this okay? Why?
            # n is even, the current z is w.l.o.g. a left endpoint of some
            # subtrees. Register the current z to be used in turning condition
            # checks later, when the right endpoints of it's subtrees are
            # generated.
            S = tree_util.tree_map(lambda arr, val: arr.at[bitcount(n)].set(val), S, z)
            return n, S, False

        def _odd_fun(n_and_S):
            n, S = n_and_S
            # n is odd, the current z is w.l.o.g a right endpoint of some
            # subtrees.  Check turning condition against all left endpoints of
            # subtrees that have the current z (/n) as their right endpoint.

            # l = nubmer of subtrees that have current z as their right endpoint.
            l = count_trailing_ones(n)
            # inclusive indices into S referring to the left endpoints of the l subtrees.
            i_max_incl = bitcount(n-1)
            i_min_incl = i_max_incl - l + 1
            # TODO: this should traverse the range in reverse
            contains_uturn = fori_loop(
                lower = i_min_incl,
                upper = i_max_incl + 1,
                # TODO: conditional for early termination
                body_fun = lambda k, contains_uturn: contains_uturn | is_euclidean_uturn_pytree(index_into_pytree_time_series(k, S), z),
                init_val = False
            )
            return n, S, contains_uturn

        _n, S, turning = cond(
            pred = n % 2 == 0,
            true_fun = _even_fun,
            false_fun = _odd_fun,
            operand = (n, S)
        )
        return (n+1, turning, chosen, z, S, key)

    _final_n, turning, chosen, _z, _S, _key = while_loop(
        # while n < 2**depth and not stop
        cond_fun=lambda state: (state[0] < 2**depth) & (~state[1]),
        body_fun=_loop_body,
        init_val=(0, False, chosen, z, S, key)
    )

    # TODO: remove this and set chosen.turning inside the loop, or: make loop state essentially just (n, chosen)
    return Tree(
        left = chosen.left,
        right = chosen.right,
        weight = chosen.weight,
        proposal_candidate = chosen.proposal_candidate,
        turning = turning
    )


def add_single_qp_to_tree(key, tree, qp, go_right, potential_energy, kinetic_energy):
    """Helper function for progressive sampling. Takes a tree with a sample, and
    a new endpoint, propagates sample. It's functional, i.e. does not modify
    arguments."""
    # This is technically just a special case of merge_trees with one of the
    # trees being a singleton, depth 0 tree.
    # TODO: just construct the singleton tree and call merge_trees
    left, right = cond(
        pred = go_right,
        true_fun = lambda tree_and_qp: (tree_and_qp[0].left, tree_and_qp[1]),
        false_fun = lambda tree_and_qp: (tree_and_qp[1], tree_and_qp[0].right),
        operand = (tree, qp)
    )
    key, subkey = random.split(key)
    total_weight = tree.weight + np.exp(-total_energy_of_qp(qp, potential_energy, kinetic_energy))
    prob_of_keeping_old = tree.weight / total_weight
    proposal_candidate = cond(
        pred = random.bernoulli(subkey, prob_of_keeping_old),
        true_fun = lambda old_and_new: old_and_new[0],
        false_fun = lambda old_and_new: old_and_new[1],
        operand = (tree.proposal_candidate, qp)
    )
    return Tree(left, right, total_weight, proposal_candidate, tree.turning)


# TOOD: remove or mark legacy
def make_tree_from_list(key, qp_of_pytree_of_series, go_right, potential_energy, kinetic_energy, turning_hint):
    # WARNING: only to be called from extend_tree_iterative, with turning_hint logic correct
    # 3. random.choice with probability weights to get sample
    #new_subtree_energies = lax.map(potential_energy, chosen_array[:,0,:]) + lax.map(kinetic_energy, chosen_array[:,1,:])
    new_subtree_energies = (
        lax.map(potential_energy, qp_of_pytree_of_series.position)
        + lax.map(kinetic_energy, qp_of_pytree_of_series.momentum)
    )
    print(f"new_subtree_energies.shape: {new_subtree_energies.shape}")
    # proportianal to the joint probabilities:
    new_subtree_weights = np.exp(-new_subtree_energies)
    # unfortunately choice only works with 1d arrays so we need to use indexing TODO: factor out?
    # complicated way of retrieving this value, maybe just pass it into the function as an argument
    number_of_samples_in_trajectory = tree_util.tree_flatten(tree_util.tree_map(
        lambda arr: arr.shape[0],
        qp_of_pytree_of_series
    ))[0][0]
    random_idx = random.choice(key, number_of_samples_in_trajectory, p=new_subtree_weights)
    new_subtree_sample = tree_util.tree_map(lambda arr: arr[random_idx], qp_of_pytree_of_series)
    print(f"chose sample n° {random_idx}")
    # 4. calculate total weight
    new_subtree_total_weight = np.sum(new_subtree_weights)
    left, right = cond(
        pred = go_right,
        true_fun = lambda first_and_last: (first_and_last[0], first_and_last[1]),
        false_fun = lambda first_and_last: (first_and_last[-1], first_and_last[1]),
        operand = (
            index_into_pytree_time_series(0, qp_of_pytree_of_series),
            index_into_pytree_time_series(-1, qp_of_pytree_of_series)
        )
    )
    return Tree(left=left, right=right, weight=new_subtree_total_weight, proposal_candidate=new_subtree_sample, turning=turning_hint)


def merge_trees(key, current_subtree, new_subtree, go_right, turning_hint):
    """Merges two trees, propagating the proposal_candidate"""
    # WARNING: only to be called from extend_tree_iterative, with turning_hint logic correct
    # 5. decide which sample to take based on total weights (merge trees)
    key, subkey = random.split(key)
    print(f"prob of choosing new sample: {new_subtree.weight / (new_subtree.weight + current_subtree.weight)}")
    new_sample = cond(
        pred = random.bernoulli(subkey, new_subtree.weight / (new_subtree.weight + current_subtree.weight)),
        # choose the new sample
        true_fun = lambda current_and_new_tup: current_and_new_tup[1],
        # choose the old sample
        false_fun = lambda current_and_new_tup: current_and_new_tup[0],
        operand = (current_subtree.proposal_candidate, new_subtree.proposal_candidate)
    )
    # 6. define new tree
    left, right = cond(
        pred = go_right,
        true_fun = lambda op: (op['current_subtree'].left, op['new_subtree'].right),
        false_fun = lambda op: (op['new_subtree'].left, op['current_subtree'].right),
        operand = {'current_subtree': current_subtree, 'new_subtree': new_subtree}
    )
    merged_tree = Tree(left=left, right=right, weight=new_subtree.weight + current_subtree.weight, proposal_candidate=new_sample, turning=turning_hint)
    return merged_tree


def bitcount(n):
    """Count the number of ones in the binary representation of n.

    Warning
    -------
    n must be positive and strictly smaller than 2**64
    
    Examples
    --------
    >>> print(bin(23), bitcount(23))
    0b10111 4
    """
    # TODO: python 3.10 has int.bit_count()
    bits_reversed = np.unpackbits(np.array(n, dtype='uint64').view('uint8'), bitorder='little')
    return np.sum(bits_reversed)


def count_trailing_ones(n):
    """Count the number of trailing, consecutive ones in the binary representation of n.

    Warning
    -------
    n must be positive and strictly smaller than 2**64

    Examples
    --------
    >>> print(bin(23), count_trailing_one_bits(23))
    0b10111 3
    """
    # taken from http://num.pyro.ai/en/stable/_modules/numpyro/infer/hmc_util.html
    _, trailing_ones_count = while_loop(
        lambda nc: (nc[0] & 1) != 0, lambda nc: (nc[0] >> 1, nc[1] + 1), (n, 0)
    )
    return trailing_ones_count


def is_euclidean_uturn(qp_left, qp_right):
    """
    See Also
    --------
    Betancourt - A conceptual introduction to Hamiltonian Monte Carlo
    """
    return (
        (np.dot(qp_right.momentum, (qp_right.position - qp_left.position)) < 0.)
        & (np.dot(qp_left.momentum, (qp_left.position - qp_right.position)) < 0.)
    )


def is_euclidean_uturn_pytree(qp_left, qp_right):
    """
    See Also
    --------
    Betancourt - A conceptual introduction to Hamiltonian Monte Carlo
    """
    # TODO: Does this work with different dtypes for different field components?
    # how does flatten_util.ravel_pytree behave in that case
    qp_left = QP(
        position=flatten_util.ravel_pytree(qp_left.position)[0],
        momentum=flatten_util.ravel_pytree(qp_left.momentum)[0]
    )
    qp_right = QP(
        position=flatten_util.ravel_pytree(qp_right.position)[0],
        momentum=flatten_util.ravel_pytree(qp_right.momentum)[0]
    )
    return is_euclidean_uturn(qp_left, qp_right)


def make_kinetic_energy_fn_from_diag_mass_matrix(mass_matrix):
    def _kin_energy(momentum):
        # calculate kinetic energies for every array (leaf) in the pytree
        kin_energies = tree_util.tree_map(lambda p, m: np.sum(p**2 / (2 * m)), momentum, mass_matrix)
        # sum everything up
        total_kin_energy = tree_util.tree_reduce(lambda acc, leaf_kin_e: acc + leaf_kin_e, kin_energies, 0.)
        return total_kin_energy
    return _kin_energy


def sample_momentum_from_diag_mass_matrix(key, diag_mass_matrix):
    key, subkey = random.split(key)
    return tree_util.tree_map(lambda m: np.sqrt(m)*random.normal(subkey, m.shape), diag_mass_matrix)


class NUTSChain:
    def __init__(self, initial_position, potential_energy, diag_mass_matrix, eps, maxdepth, rngseed):
        self.position = initial_position

        # TODO: typechecks?
        self.potential_energy = potential_energy

        if isinstance(diag_mass_matrix, float):
            self.diag_mass_matrix = tree_util.tree_map(lambda arr: np.full(arr.shape, diag_mass_matrix), initial_position)
        elif tree_util.tree_structure(diag_mass_matrix) == tree_util.tree_structure(initial_position):
            shape_match_tree = tree_util.tree_map(lambda a1, a2: a1.shape == a2.shape, diag_mass_matrix, initial_position)
            shape_and_structure_match = all(tree_util.tree_flatten(shape_match_tree))
            if shape_and_structure_match:
                self.diag_mass_matrix = diag_mass_matrix
            else:
                raise ValueError("diag_mass_matrix has same tree_structe as initial_position but shapes don't match up")
        else:
            raise ValueError('diag_mass_matrix must either be float or have same tree structure as initial_position')

        if isinstance(eps, float):
            self.eps = eps
        else:
            raise ValueError('eps must be a float')

        if isinstance(maxdepth, int):
            self.maxdepth = maxdepth
        else:
            raise ValueError('maxdepth must be an int')
        
        self.key = random.PRNGKey(rngseed)


    def generate_n_samples(self, n):
        potential_energy_gradient = grad(self.potential_energy)
        stepper = lambda qp, eps, direction: leapfrog_step_pytree(potential_energy_gradient, qp, eps*direction)[0]

        samples = tree_util.tree_map(lambda arr: np.ones((n,) + arr.shape), self.position)

        def _body_fun(idx, state):
            prev_position, key, samples = state
            key, subkey = random.split(key)
            resampled_momentum = sample_momentum_from_diag_mass_matrix(subkey, self.diag_mass_matrix)

            qp = QP(position=prev_position, momentum=resampled_momentum)

            key, subkey = random.split(key)
            tree = generate_nuts_sample(
                initial_qp = qp,
                key = subkey,
                eps = self.eps,
                maxdepth = self.maxdepth,
                stepper = stepper,
                potential_energy = self.potential_energy,
                kinetic_energy = make_kinetic_energy_fn_from_diag_mass_matrix(self.diag_mass_matrix)
            )
            print("current sample", tree.proposal_candidate)
            samples = tree_util.tree_map(lambda ts, val: ts.at[idx].set(val), samples, tree.proposal_candidate.position)
            return (tree.proposal_candidate.position, key, samples)
        
        jitted = jit(lambda : fori_loop(lower=0, upper=n, body_fun=_body_fun, init_val=(self.position, self.key, samples)))
        return jitted()
