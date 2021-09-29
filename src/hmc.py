from collections import namedtuple
from jax import numpy as np
from jax import tree_util
from jax import lax, random, jit, partial, flatten_util, grad
from jax.scipy.special import expit

from .disable_jax_control_flow import cond, while_loop, fori_loop
from .sugar import random_like

_DEBUG_FLAG = False

from jax.experimental import host_callback

_DEBUG_TREE_END_IDXS = []
_DEBUG_SUBTREE_END_IDXS = []
_DEBUG_STORE = []

def _DEBUG_ADD_QP(qp):
    """Stores **all** results of leapfrog integration"""
    global _DEBUG_STORE
    _DEBUG_STORE.append(qp)

def _DEBUG_FINISH_TREE(dummy_arg):
    """Signal the position of a finished tree in `_DEBUG_STORE`"""
    global _DEBUG_TREE_END_IDXS
    _DEBUG_TREE_END_IDXS.append(len(_DEBUG_STORE))

def _DEBUG_FINISH_SUBTREE(dummy_arg):
    """Signal the position of a finished sub-tree in `_DEBUG_STORE`"""
    global _DEBUG_SUBTREE_END_IDXS
    _DEBUG_SUBTREE_END_IDXS.append(len(_DEBUG_STORE))


###
### COMMON FUNCTIONALITY
###

# A datatype for (q, p) = (position, momentum) pairs
QP = namedtuple('QP', ['position', 'momentum'])


def flip_momentum(qp: QP) -> QP:
    return QP(position=qp.position, momentum=-qp.momentum)


def sample_momentum_from_diagonal(*, key, diag_mass_matrix):
    """
    Draw a momentum sample from the kinetic energy of the hamiltonian.

    Parameters
    ----------
    key: ndarray
        a PRNGKey used as the random key.
    diag_mass_matrix: ndarray
        The mass matrix (i.e. inverse diagonal covariance) to use for sampling.
        Diagonal matrix represented as (possibly pytree of) ndarray vector
        containing the entries of the diagonal.
    """
    normal = random_like(diag_mass_matrix, key=key, rng=random.normal)
    return tree_util.tree_map(lambda m, nrm: np.sqrt(m) * nrm, diag_mass_matrix, normal)


# TODO: how to randomize step size (neal sect. 3.2)
# @partial(jit, static_argnames=('potential_energy_gradient',))
def leapfrog_step(
        qp: QP,
        potential_energy_gradient,
        step_length,
        mass_matrix,
    ):
    """
    Perform one iteration of the leapfrog integrator forwards in time.

    Parameters
    ----------
    potential_energy_gradient: Callable[[ndarray], float]
        Potential energy gradient part of the hamiltonian (V). Depends on position only.
    qp: QP
        Point in position and momentum space from which to start integration.
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

    position_fullstep = position + step_length * momentum_halfstep / mass_matrix # type: ignore
    #print("position_fullstep:", position_fullstep)

    momentum_fullstep = (
        momentum_halfstep
        - (step_length / 2.) * potential_energy_gradient(position_fullstep)  # type: ignore
    )
    #print("momentum_fullstep:", momentum_fullstep)

    qp_fullstep = QP(position=position_fullstep, momentum=momentum_fullstep)
    return qp_fullstep


def leapfrog_step_pytree(
    qp: QP,
    potential_energy_gradient,
    step_length,
    mass_matrix
    ) -> QP:
    position = qp.position
    momentum = qp.momentum

    momentum_halfstep = tree_util.tree_map(
        lambda mom, potgrad: mom - (step_length / 2.) * potgrad,
        momentum,
        potential_energy_gradient(position)
    )

    position_fullstep = tree_util.tree_map(
        lambda pos, mom_halfstep, mass: pos + step_length * mom_halfstep / mass,  # type: ignore
        position,
        momentum_halfstep,
        mass_matrix
    )

    momentum_fullstep = tree_util.tree_map(
        lambda mom_halfstep, potgrad: mom_halfstep - (step_length / 2.) * potgrad,
        momentum_halfstep,
        potential_energy_gradient(position_fullstep)
    )

    qp_fullstep = QP(position=position_fullstep, momentum=momentum_fullstep)

    global _DEBUG_FLAG
    if _DEBUG_FLAG:
        # append result to global list variable
        host_callback.call(_DEBUG_ADD_QP, qp_fullstep)

    return qp_fullstep


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
    acceptance_threshold = np.minimum(
            1.,
            np.exp(
                total_energy(old_qp)
                - total_energy(proposed_qp)
            )
        )

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
        # TODO remove this parameter, instead rework this to take a `stepper` function just like `generate_nuts_sample` and have the mass matrix there.
        mass_matrix,
        kinetic_energy,
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
    mass_matrix: ndarray
        The mass matrix used in the kinetic energy
    number_of_integration_steps: int
        The number of steps the leapfrog integrator should perform.
    step_length: float
        The step size (usually epsilon) for the leapfrog integrator.
    """
    key, subkey = random.split(key)
    momentum = sample_momentum_from_diagonal(
        key = subkey,
        diag_mass_matrix = mass_matrix
    )
    qp = QP(position=position, momentum=momentum)

    loop_body = partial(leapfrog_step, potential_energy_gradient=potential_energy_gradient, step_length=step_length, mass_matrix=mass_matrix)
    new_qp = fori_loop(
        lower = 0,
        upper = number_of_integration_steps,
        body_fun = lambda _, args: loop_body(args),
        init_val = qp
    )

    # this flipping is needed to make the proposal distribution symmetric
    # doesn't have any effect on acceptance though because kinetic energy depends on momentum^2
    # might have an effect with other kinetic energies though
    proposed_qp = flip_momentum(new_qp)

    return accept_or_deny(
        key = key,
        old_qp = qp,
        proposed_qp = proposed_qp,
        total_energy = lambda qp: total_energy_of_qp(qp, potential_energy, kinetic_energy)
    ), momentum


###
### NUTS
###

# A datatype carrying tree metadata
# left, right: endpoints of the trees path
# logweight: sum over all -H(q, p) in the trees path
# proposal_candidate: random sample from the trees path, distributed as exp(-H(q, p))
# turning: TODO: whole tree or also subtrees??
# turning currently means either the left, right endpoint are a uturn or any subtree is a uturn, see TODO above
Tree = namedtuple('Tree', ['left', 'right', 'logweight', 'proposal_candidate', 'turning', 'depth'])


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
    current_tree = Tree(left=initial_qp, right=initial_qp, logweight=-total_energy_of_qp(initial_qp, potential_energy, kinetic_energy), proposal_candidate=initial_qp, turning=False, depth=0)

    # loop stopping condition
    stop = False
    # loop tree depth, increases each iteration
    j = 0

    loop_state = (key, current_tree, j, stop)

    def _cond_fn(loop_state):
        _key, _current_tree, j, stop = loop_state
        # while (not stop) and j <= maxdepth
        return (~stop) & (j <= maxdepth)

    def _body_fun(loop_state):
        key, current_tree, j, stop = loop_state
        key, key_dir, key_subtree, key_merge = random.split(key, 4)

        go_right = random.bernoulli(key_dir, 0.5)

        # build tree of depth j, adjacent to current_tree
        new_subtree = iterative_build_tree(key_subtree, current_tree, eps, go_right, stepper, potential_energy, kinetic_energy, maxdepth)

        # combine current_tree and new_subtree into a depth j+1 tree only if new_subtree has no turning subtrees (including itself)
        current_tree = cond(
            pred = new_subtree.turning,
            true_fun = lambda old_and_new: old_and_new[0],
            false_fun = lambda old_and_new: merge_trees(key_merge, old_and_new[0], old_and_new[1], go_right),
            operand = (current_tree, new_subtree),
        )
        # stop if new subtree was turning -> we sample from the old one and don't expand further
        # stop if new total tree is turning -> we sample from the combined trajectory and don't expand further
        stop = new_subtree.turning | current_tree.turning
        j = j + 1
        return (key, current_tree, j, stop)

    _key, current_tree, _j, _stop = while_loop(_cond_fn, _body_fun, loop_state)

    global _DEBUG_FLAG
    if _DEBUG_FLAG:
        host_callback.call(_DEBUG_FINISH_TREE, None)

    return current_tree


def index_into_pytree_time_series(idx, ptree):
    return tree_util.tree_map(lambda arr: arr[idx], ptree)


# Essentially algorithm 2 from https://arxiv.org/pdf/1912.11554.pdf
def iterative_build_tree(key, initial_tree, eps, go_right, stepper, potential_energy, kinetic_energy, maxdepth):
    """
    Starting from either the left or right endpoint of a given tree, builds a new adjacent tree of the same size.

    Parameters
    ----------
    key: ndarray
        randomness uses to choose a sample when adding QPs to the tree
    initial_tree: Tree
        Tree to be extended (doubled) on the left or right.
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
        It's only required to statically set the size of the `S` array (pytree).
    """
    # 1. choose start point of integration
    # TODO: Use pytree-enabled select
    initial_qp = cond(
        pred = go_right,
        true_fun = lambda left_and_right: left_and_right[1],
        false_fun = lambda left_and_right: left_and_right[0],
        operand = (initial_tree.left, initial_tree.right)
    )
    z = initial_qp
    depth = initial_tree.depth
    # 2. build / collect chosen states
    # TODO: rename chosen to a more sensible name such as new_tree ...
    # TODO: WARNING: this will be overwritten in the first iteration of the loop, the assignment to chosen is only temporary and we're using z since it's the only QP that's availible right now. This would also be solved by moving the first iteration outside of the loop.
    # TODO: maybe chosen = None works? Or maybe it's not required at all?
    chosen = Tree(z,z,0.,z,turning=False,depth=depth)
    # Storage for left endpoints of subtrees. Size is determined statically by the `maxdepth` parameter.
    # NOTE, let's hope this does not break anything but in principle we only
    # need `maxdepth` element even though the tree can be of length `maxdepth +
    # 1` since we will never access the last element...
    S = tree_util.tree_map(lambda initial_q_or_p_leaf: np.empty((maxdepth, ) + initial_q_or_p_leaf.shape), unzip_qp_pytree(initial_qp))

    def _loop_body(state):
        n, _turning, chosen, z, S, key = state

        z = stepper(z, eps, np.where(go_right, x=1, y=-1))

        key, subkey = random.split(key)
        # TODO: maybe just move the first iteration outside of the loop?
        chosen = cond(
            pred = n == 0,
            # first iteration: create the depth 0 tree
            true_fun = lambda c_and_z: Tree(left=z, right=z, logweight=-total_energy_of_qp(z, potential_energy, kinetic_energy), proposal_candidate=z, turning=False, depth=depth),
            # all later iterations: add new point to `chosen` tree
            false_fun = lambda c_and_z: add_single_qp_to_tree(subkey, chosen, z, go_right, potential_energy, kinetic_energy),
            operand = (chosen, z)
        )

        def _even_fun(S):
            # n is even, the current z is w.l.o.g. a left endpoint of some
            # subtrees. Register the current z to be used in turning condition
            # checks later, when the right endpoints of it's subtrees are
            # generated.
            S = tree_util.tree_map(lambda arr, val: arr.at[bitcount(n)].set(val), S, z)
            return S, False

        def _odd_fun(S):
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
            return S, contains_uturn

        S, turning = cond(
            pred = n % 2 == 0,
            true_fun = _even_fun,
            false_fun = _odd_fun,
            operand = S
        )
        return (n+1, turning, chosen, z, S, key)

    _final_n, turning, chosen, _z, _S, _key = while_loop(
        # while n < 2**depth and not stop
        cond_fun=lambda state: (state[0] < 2**depth) & (~state[1]),
        body_fun=_loop_body,
        init_val=(0, False, chosen, z, S, key)
    )

    global _DEBUG_FLAG
    if _DEBUG_FLAG:
        host_callback.call(_DEBUG_FINISH_SUBTREE, None)

    # TODO: remove this and set chosen.turning inside the loop, or: make loop state essentially just (n, chosen)
    return Tree(
        left = chosen.left,
        right = chosen.right,
        logweight = chosen.logweight,
        proposal_candidate = chosen.proposal_candidate,
        turning = turning,
        depth = chosen.depth
    )


def add_single_qp_to_tree(key, tree, qp, go_right, potential_energy, kinetic_energy):
    """Helper function for progressive sampling. Takes a tree with a sample, and
    a new endpoint, propagates sample. It's functional, i.e. does not modify
    arguments."""
    # This is technically just a special case of merge_trees with one of the
    # trees being a singleton, depth 0 tree.
    # TODO: just construct the singleton tree and call merge_trees
    # TODO: Use pytree-enabled select
    left, right = cond(
        pred = go_right,
        true_fun = lambda tree_and_qp: (tree_and_qp[0].left, tree_and_qp[1]),
        false_fun = lambda tree_and_qp: (tree_and_qp[1], tree_and_qp[0].right),
        operand = (tree, qp)
    )
    qp_logweight = -total_energy_of_qp(qp, potential_energy, kinetic_energy)
    # ln(e^-H_1 + e^-H_2)
    total_logweight = np.logaddexp(tree.logweight, qp_logweight)
    # expit(x-y) := 1 / (1 + e^(-(x-y))) = 1 / (1 + e^(y-x)) = e^x / (e^y + e^x)
    prob_of_keeping_old = expit(tree.logweight - qp_logweight)
    # TODO: Use pytree-enabled select
    proposal_candidate = cond(
        pred = random.bernoulli(key, prob_of_keeping_old),
        true_fun = lambda old_and_new: old_and_new[0],
        false_fun = lambda old_and_new: old_and_new[1],
        operand = (tree.proposal_candidate, qp)
    )
    return Tree(left, right, total_logweight, proposal_candidate, tree.turning, tree.depth)


def merge_trees(key, current_subtree, new_subtree, go_right):
    """Merges two trees, propagating the proposal_candidate"""
    # 5. decide which sample to take based on total weights (merge trees)
    key, subkey = random.split(key)
    # expit(x-y) := 1 / (1 + e^(-(x-y))) = 1 / (1 + e^(y-x)) = e^x / (e^y + e^x)
    prob_of_choosing_new = expit(new_subtree.logweight - current_subtree.logweight)
    print(f"prob of choosing new sample: {prob_of_choosing_new}")
    # NOTE, here it is possible to bias the transition towards the new subtree
    # Betancourt cenceptual intro (and Numpyro)
    # TODO: Use pytree-enabled select
    new_sample = cond(
        pred = random.bernoulli(subkey, prob_of_choosing_new),
        # choose the new sample
        true_fun = lambda current_and_new_tup: current_and_new_tup[1],
        # choose the old sample
        false_fun = lambda current_and_new_tup: current_and_new_tup[0],
        operand = (current_subtree.proposal_candidate, new_subtree.proposal_candidate)
    )
    # 6. define new tree
    # TODO: Use pytree-enabled select
    left, right = cond(
        pred = go_right,
        true_fun = lambda op: (op['current_subtree'].left, op['new_subtree'].right),
        false_fun = lambda op: (op['new_subtree'].left, op['current_subtree'].right),
        operand = {'current_subtree': current_subtree, 'new_subtree': new_subtree}
    )
    turning = is_euclidean_uturn_pytree(left, right)
    merged_tree = Tree(left=left, right=right, logweight=np.logaddexp(new_subtree.logweight, current_subtree.logweight), proposal_candidate=new_sample, turning=turning, depth=current_subtree.depth + 1)
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


# TODO: implement directly in is_euclidean_uturn, don't use ravel_pytree but tree_map.
# Use np.dot but make sure to ravel arrays to avoid matrix multiplication
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


class NUTSChain:
    def __init__(self, initial_position, potential_energy, diag_mass_matrix, eps, maxdepth, rngseed, compile=True, dbg_info=False, signal_response=lambda x: x):
        self.position = initial_position

        # TODO: typechecks?
        self.potential_energy = potential_energy

        #if not diag_mass_matrix == 1.:
        #    raise NotImplementedError("Leapfrog integrator doesn't support custom mass matrix yet.")

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

        potential_energy_gradient = grad(self.potential_energy)
        self.stepper = lambda qp, eps, direction: leapfrog_step_pytree(qp, potential_energy_gradient, eps*direction, self.diag_mass_matrix)

        if isinstance(maxdepth, int):
            self.maxdepth = maxdepth
        else:
            raise ValueError('maxdepth must be an int')

        self.key = random.PRNGKey(rngseed)

        self.compile = compile

        self.dbg_info = dbg_info

        self.signal_response = signal_response


    def generate_n_samples(self, n):

        samples = tree_util.tree_map(lambda arr: np.ones((n,) + arr.shape), self.position)

        if self.dbg_info:
            momenta_before = tree_util.tree_map(lambda arr: np.ones((n,) + arr.shape), self.position)
            momenta_after = tree_util.tree_map(lambda arr: np.ones((n,) + arr.shape), self.position)
            depths = np.empty(n, dtype=np.int8)
            # just a prototype qp
            _qp_proto = QP(self.position, self.position)
            # just a prototype tree
            _tree_proto = Tree(_qp_proto, _qp_proto, 0., _qp_proto, True, 0)
            trees = tree_util.tree_map(
                lambda leaf: np.empty_like(leaf, shape=(n,)+np.array(leaf).shape),
                _tree_proto

            )

        def _body_fun(idx, state):
            if self.dbg_info:
                prev_position, key, samples, momenta_before, momenta_after, depths, trees = state
            else:
                prev_position, key, samples = state
            key, key_momentum, key_nuts = random.split(key, 3)

            resampled_momentum = sample_momentum_from_diagonal(
                key=key_momentum,
                diag_mass_matrix=self.diag_mass_matrix
            )

            qp = QP(position=prev_position, momentum=resampled_momentum)

            tree = generate_nuts_sample(
                initial_qp = qp,
                key = key_nuts,
                eps = self.eps,
                maxdepth = self.maxdepth,
                stepper = self.stepper,
                potential_energy = self.potential_energy,
                kinetic_energy = make_kinetic_energy_fn_from_diag_mass_matrix(self.diag_mass_matrix)
            )
            #print("current sample", tree.proposal_candidate)
            samples = tree_util.tree_map(lambda ts, val: ts.at[idx].set(val), samples, tree.proposal_candidate.position)
            if self.dbg_info:
                momenta_before = tree_util.tree_map(lambda ts, val: ts.at[idx].set(val), momenta_before, resampled_momentum)
                momenta_after = tree_util.tree_map(lambda ts, val: ts.at[idx].set(val), momenta_after, tree.proposal_candidate.momentum)
                depths = depths.at[idx].set(tree.depth)
                trees = tree_util.tree_map(lambda ts, val: ts.at[idx].set(val), trees, tree)

            updated_state = (tree.proposal_candidate.position, key, samples)

            if self.dbg_info:
                updated_state = updated_state + (momenta_before, momenta_after, depths, trees)

            return updated_state

        loop_initial_state = (self.position, self.key, samples)
        if self.dbg_info:
            loop_initial_state = loop_initial_state + (momenta_before, momenta_after, depths, trees)

        return_fn = lambda: fori_loop(lower=0, upper=n, body_fun=_body_fun, init_val=loop_initial_state)

        if self.compile:
            results = jit(return_fn)()
        else:
            results = return_fn()

        names = ('position', 'key', 'samples')
        if self.dbg_info:
            names = names + ('momenta_before', 'momenta_after', 'depths', 'trees')

        self.results = {k: v for (k, v) in zip(names, results)}
        self.results['response'] = lax.map(self.signal_response, self.results['samples'])

        return results


    def plot_1d_sample_mean(self, ax, **kwargs):
        response = self.results['response']
        kwargs['label'] = kwargs.get('label', 'mean of signal response of samples')
        if len(response.shape) == 2:
            resp_mean = np.mean(response, axis=0)
            ax.plot(resp_mean, **kwargs)
        else:
            raise NotImplementedError


    def plot_response_ts(self, ax, **kwargs):
        response = self.results['response']
        if len(response.shape) == 1:
            ax.plot(response, **kwargs)
        else:
            raise NotImplementedError


    def plot_1d_hist(self, ax, **kwargs):
        response = self.results['response']
        if len(response.shape) != 1:
            raise NotImplementedError
        plot_prob = kwargs.pop('plot_prob', False)
        _, bins, _ = ax.hist(response, density=kwargs.pop('density', plot_prob), **kwargs)
        if plot_prob:
            y = np.exp(-self.potential_energy(bins))
            Z = np.trapz(y, bins)
            if not np.isfinite(Z):
                raise RuntimeError
            y = y / Z
            ax.plot(bins, y, label='probability density')
            ax.legend()


    def plot_ham_ts(self, ax, **kwargs):
        xlabel = kwargs.pop('xlabel', 'iteration number')
        title = kwargs.pop('title', 'potential energy time series')
        samples = self.results['samples']
        ham_ts = lax.map(self.potential_energy, samples)
        ax.plot(ham_ts, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_title(title)


    def plot_depth_hist(self, ax, **kwargs):
        xlabel = kwargs.pop('xlabel', 'depth')
        title = kwargs.pop('title', 'tree depth histogram')
        depths = self.results['depths']
        bins = np.arange(1, depths.max() + 1.5) - 0.5
        ax.hist(depths, bins, **kwargs)
        ax.set_xticks(bins+0.5)
        ax.set_xlabel(xlabel)
        ax.set_title(title)


class HMCChain:
    def __init__(self, initial_position, potential_energy, diag_mass_matrix, eps, n_of_integration_steps, rngseed, compile=True, dbg_info=False):
        self.position = initial_position

        # TODO: typechecks?
        self.potential_energy = potential_energy

        if not diag_mass_matrix == 1.:
            # TODO: check diagonal_momentum_covariance name and implementaiton in accetpance and such
            raise NotImplementedError("Leapfrog integrator doesn't support custom mass matrix yet.")

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

        if isinstance(n_of_integration_steps, int):
            self.n_of_integration_steps = n_of_integration_steps
        else:
            raise ValueError('n_of_integration_steps must be an int')

        self.key = random.PRNGKey(rngseed)

        self.compile = compile

        self.dbg_info = dbg_info

    def generate_n_samples(self, n):

        samples = tree_util.tree_map(lambda arr: np.ones((n,) + arr.shape), self.position)
        acceptance = np.empty(n, dtype=bool)

        if self.dbg_info:
            momenta_before = tree_util.tree_map(lambda arr: np.ones((n,) + arr.shape), self.position)
            momenta_after = tree_util.tree_map(lambda arr: np.ones((n,) + arr.shape), self.position)
            rejected_position_samples = tree_util.tree_map(lambda arr: np.ones((n,) + arr.shape), self.position)
            rejected_momenta = tree_util.tree_map(lambda arr: np.ones((n,) + arr.shape), self.position)

        def _body_fun(idx, state):
            if self.dbg_info:
                prev_position, key, samples, acceptance, momenta_before, momenta_after, rejected_position_samples, rejected_momenta = state
            else:
                prev_position, key, samples, acceptance = state
            key, key_hmc = random.split(key)

            (qp_old_and_proposed_sample, was_accepted), unintegrated_momentum = generate_hmc_sample(
                key = key_hmc,
                position = prev_position,
                potential_energy = self.potential_energy,
                potential_energy_gradient = grad(self.potential_energy),
                mass_matrix = self.diag_mass_matrix,
                kinetic_energy = make_kinetic_energy_fn_from_diag_mass_matrix(self.diag_mass_matrix),
                number_of_integration_steps = self.n_of_integration_steps,
                step_length = self.eps
            )

            # TODO: what to do with the other one (it's rejected or just the previous sample in case the new one was accepted)
            # TODO: Use pytree-enabled select
            next_qp, rejected_qp = cond(
                pred = was_accepted,
                true_fun = lambda tup: (tup[1], tup[0]),
                false_fun = lambda tup: (tup[0], tup[1]),
                operand = qp_old_and_proposed_sample
            )

            samples = tree_util.tree_map(lambda ts, val: ts.at[idx].set(val), samples, next_qp.position)
            acceptance = acceptance.at[idx].set(was_accepted)
            if self.dbg_info:
                momenta_before = tree_util.tree_map(lambda ts, val: ts.at[idx].set(val), momenta_before, unintegrated_momentum)
                momenta_after = tree_util.tree_map(lambda ts, val: ts.at[idx].set(val), momenta_after, next_qp.momentum)
                rejected_position_samples = tree_util.tree_map(lambda ts, val: ts.at[idx].set(val), rejected_position_samples, rejected_qp.position)
                rejected_momenta = tree_util.tree_map(lambda ts, val: ts.at[idx].set(val), rejected_position_samples, rejected_qp.momentum)

            updated_state = (next_qp.position, key, samples, acceptance)

            if self.dbg_info:
                updated_state = updated_state + (momenta_before, momenta_after, rejected_position_samples, rejected_momenta)

            return updated_state

        loop_initial_state = (self.position, self.key, samples, acceptance)
        if self.dbg_info:
            loop_initial_state = loop_initial_state + (momenta_before, momenta_after, rejected_position_samples, rejected_momenta)

        return_fn = lambda: fori_loop(lower=0, upper=n, body_fun=_body_fun, init_val=loop_initial_state)
        if self.compile:
            return jit(return_fn)()
        else:
            return return_fn()
