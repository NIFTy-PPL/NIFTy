from functools import partial
from jax import numpy as np
from jax import tree_util
from jax import lax, random, jit, grad
from jax.scipy.special import expit
from jax.lax import population_count

from typing import Any, Callable, NamedTuple, Optional, TypeVar, Union

from .disable_jax_control_flow import cond, while_loop, fori_loop
from .sugar import random_like
from .forest_util import select


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

Q = TypeVar("Q")

class QP(NamedTuple):
    """Object holding a pair of position and momentum.

    Attributes
    ----------
    position : Q
        Position.
    momentum : Q
        Momentum.
    """
    position: Q
    momentum: Q


def flip_momentum(qp: QP) -> QP:
    return QP(position=qp.position, momentum=-qp.momentum)


def sample_momentum_from_diagonal(*, key, mass_matrix_sqrt):
    """
    Draw a momentum sample from the kinetic energy of the hamiltonian.

    Parameters
    ----------
    key: ndarray
        a PRNGKey used as the random key.
    mass_matrix_sqrt: ndarray
        The left square-root mass matrix (i.e. square-root of the inverse
        diagonal covariance) to use for sampling. Diagonal matrix represented
        as (possibly pytree of) ndarray vector containing the entries of the
        diagonal.
    """
    normal = random_like(mass_matrix_sqrt, key=key, rng=random.normal)
    return tree_util.tree_map(np.multiply, mass_matrix_sqrt, normal)


# TODO: how to randomize step size (neal sect. 3.2)
# @partial(jit, static_argnames=('potential_energy_gradient',))
def leapfrog_step(
        potential_energy_gradient,
        kinetic_energy_gradient,
        step_size,
        inverse_mass_matrix,
        qp: QP,
    ):
    """
    Perform one iteration of the leapfrog integrator forwards in time.

    Parameters
    ----------
    potential_energy_gradient: Callable[[ndarray], float]
        Potential energy gradient part of the hamiltonian (V). Depends on position only.
    qp: QP
        Point in position and momentum space from which to start integration.
    step_size: float
        Step length (usually called epsilon) of the leapfrog integrator.
    """
    position = qp.position
    momentum = qp.momentum

    momentum_halfstep = (
        momentum
        - (step_size / 2.) * potential_energy_gradient(position)
    )

    position_fullstep = position + step_size * kinetic_energy_gradient(inverse_mass_matrix, momentum_halfstep)

    momentum_fullstep = (
        momentum_halfstep
        - (step_size / 2.) * potential_energy_gradient(position_fullstep)
    )

    qp_fullstep = QP(position=position_fullstep, momentum=momentum_fullstep)

    global _DEBUG_FLAG
    if _DEBUG_FLAG:
        # append result to global list variable
        host_callback.call(_DEBUG_ADD_QP, qp_fullstep)

    return qp_fullstep


class AcceptedAndRejected(NamedTuple):
    accepted_qp: QP
    rejected_qp: QP
    accepted: Union[np.ndarray, bool]
    diverging: Union[np.ndarray, bool]


###
### SIMPLE HMC
###
# @partial(jit, static_argnames=('potential_energy', 'potential_energy_gradient'))
def _generate_hmc_acc_rej(*,
        key,
        initial_qp,
        potential_energy,
        kinetic_energy,
        inverse_mass_matrix,
        stepper,
        num_steps,
        step_size,
        max_energy_difference
    ) -> AcceptedAndRejected:
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
    num_steps: int
        The number of steps the leapfrog integrator should perform.
    step_size: float
        The step size (usually epsilon) for the leapfrog integrator.
    """
    loop_body = partial(stepper, step_size, inverse_mass_matrix)
    new_qp = fori_loop(
        lower = 0,
        upper = num_steps,
        body_fun = lambda _, args: loop_body(args),
        init_val = initial_qp
    )
    # this flipping is needed to make the proposal distribution symmetric
    # doesn't have any effect on acceptance though because kinetic energy depends on momentum^2
    # might have an effect with other kinetic energies though
    proposed_qp = flip_momentum(new_qp)

    # TODO: new energy quickly becomes NaN, can be fixed by keeping step size small (?)
    # how to handle this case?
    # TODO: swap nan as energy difference with inf energy
    total_energy = partial(
        total_energy_of_qp,
        potential_energy=potential_energy,
        kinetic_energy_w_inv_mass=partial(kinetic_energy, inverse_mass_matrix)
    )
    energy_diff = total_energy(initial_qp) - total_energy(proposed_qp)
    transition_probability = np.minimum(1., np.exp(energy_diff))
    # TODO: Use bernoulli
    acceptance_level = random.uniform(key)

    accept = acceptance_level < transition_probability
    accepted_qp, rejected_qp = select(
        accept,
        (proposed_qp, initial_qp),
        (initial_qp, proposed_qp),
    )
    diverging = np.abs(energy_diff) > max_energy_difference
    return AcceptedAndRejected(accepted_qp, rejected_qp, accepted=accept, diverging=diverging)


###
### NUTS
###
class Tree(NamedTuple):
    """Object carrying tree metadata.

    Attributes
    ----------
    left, right : QP
        Respective endpoints of the trees path.
    logweight: Union[np.ndarray, float]
        Sum over all -H(q, p) in the tree's path.
    proposal_candidate: QP
        Sample from the trees path, distributed as exp(-H(q, p)).
    turning: Union[np.ndarray, bool]
        Indicator for either the left or right endpoint are a uturn or any
        subtree is a uturn.
    diverging: Union[np.ndarray, bool]
        Indicator for a large increase in energy.
    depth: Union[np.ndarray, int]
        Levels of the tree.
    """
    left: QP
    right: QP
    logweight: Union[np.ndarray, float]
    proposal_candidate: QP
    turning: Union[np.ndarray, bool]
    diverging: Union[np.ndarray, bool]
    depth: Union[np.ndarray, int]

class Chain(NamedTuple):
    """Object carrying chain metadata; think: transposed Tree with new axis.
    """
    # Q but with one more dimension on the first axes of the leave tensors
    samples: Q
    divergences: np.ndarray
    depths: Optional[np.ndarray] = None
    acceptance: Union[None,np.ndarray,float] = None
    resampled_momenta: Optional[Q] = None
    trees: Optional[Union[Tree,AcceptedAndRejected]] = None



def total_energy_of_qp(qp, potential_energy, kinetic_energy_w_inv_mass):
    return potential_energy(qp.position) + kinetic_energy_w_inv_mass(qp.momentum)


def _generate_nuts_tree(initial_qp, key, step_size, max_tree_depth, stepper: Callable[[Union[np.ndarray, float], Q, QP], QP], potential_energy, kinetic_energy: Callable[[Q, Q], float], inverse_mass_matrix: Q, bias_transition: bool=True, max_energy_difference: Union[np.ndarray, float]=np.inf) -> Tree:
    """Generate a sample given the initial position.

    This call implements a No-U-Turn-Sampler.

    Parameters
    ----------
    initial_qp: QP
        starting (position, momentum) pair
        WARNING: momentum must be resampled from conditional distribution BEFORE passing into this function!
    key: ndarray
        a PRNGKey used as the random key
    step_size: float
        The step size (usually called epsilon) for the leapfrog integrator.
    max_tree_depth: int
        The maximum depth of the trajectory tree before expansion is terminated
        and value is sampled even if the U-turn condition is not met.
        The maximum number of points (/integration steps) per trajectory is
            N = 2**max_tree_depth
        Memory requirements of this function are linear in max_tree_depth, i.e. logarithmic in trajectory length.
        JIT: static argument
    stepper: Callable[[float, Q, QP], QP]
        The function that performs (Leapfrog) steps. Takes as arguments (in order)
            step size (containing the direction): float
            inverse mass matrix: Q
            starting point: QP
    potential_energy: Callable[[Q], float]
        The potential energy, of the distribution to be sampled from.
        Takes only the position part (QP.position) as argument
    kinetic_energy: Callable[[Q, Q], float], optional
        Mapping of the momentum to its corresponding kinetic energy. As
        argument the function takes the inverse mass matrix and the momentum.

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
    initial_neg_energy = -total_energy_of_qp(initial_qp, potential_energy, partial(kinetic_energy, inverse_mass_matrix))
    current_tree = Tree(left=initial_qp, right=initial_qp, logweight=initial_neg_energy, proposal_candidate=initial_qp, turning=False, diverging=False, depth=0)

    def _cont_cond(loop_state):
        _, current_tree, stop = loop_state
        return (~stop) & (current_tree.depth <= max_tree_depth)

    def cond_tree_doubling(loop_state):
        key, current_tree, _ = loop_state
        key, key_dir, key_subtree, key_merge = random.split(key, 4)

        go_right = random.bernoulli(key_dir, 0.5)

        # build tree adjacent to current_tree
        new_subtree = iterative_build_tree(key_subtree, current_tree, step_size, go_right, stepper, potential_energy, kinetic_energy, inverse_mass_matrix, max_tree_depth, initial_neg_energy=initial_neg_energy, max_energy_difference=max_energy_difference)

        # combine current_tree and new_subtree into a tree which is one layer deeper only if new_subtree has no turning subtrees (including itself)
        current_tree = cond(
            # If new tree is turning or diverging, do not merge
            pred = new_subtree.turning | new_subtree.diverging,
            true_fun = lambda old_and_new: old_and_new[0],
            false_fun = lambda old_and_new: merge_trees(key_merge, old_and_new[0], old_and_new[1], go_right, bias_transition=bias_transition),
            operand = (current_tree, new_subtree),
        )
        # stop if new subtree was turning -> we sample from the old one and don't expand further
        # stop if new total tree is turning -> we sample from the combined trajectory and don't expand further
        stop = new_subtree.turning | current_tree.turning
        stop |= new_subtree.diverging
        return (key, current_tree, stop)

    loop_state = (key, current_tree, False)
    _, current_tree, _ = while_loop(_cont_cond, cond_tree_doubling, loop_state)

    global _DEBUG_FLAG
    if _DEBUG_FLAG:
        host_callback.call(_DEBUG_FINISH_TREE, None)

    return current_tree


def tree_index_get(ptree, idx):
    return tree_util.tree_map(lambda arr: arr[idx], ptree)


def tree_index_update(x, idx, y):
    from jax.tree_util import tree_map
    from jax.ops import index_update

    return tree_map(lambda x_el, y_el: index_update(x_el, idx, y_el), x, y)


# Essentially algorithm 2 from https://arxiv.org/pdf/1912.11554.pdf
def iterative_build_tree(key, initial_tree, step_size, go_right, stepper, potential_energy, kinetic_energy, inverse_mass_matrix, max_tree_depth, initial_neg_energy, max_energy_difference):
    """
    Starting from either the left or right endpoint of a given tree, builds a new adjacent tree of the same size.

    Parameters
    ----------
    key: ndarray
        randomness uses to choose a sample when adding QPs to the tree
    initial_tree: Tree
        Tree to be extended (doubled) on the left or right.
    step_size: float
        The step size (usually called epsilon) for the leapfrog integrator.
    go_right: bool
        If go_right start at the right end, going right else start at the left end, going left.
    stepper: Callable[[float, Q, QP], QP]
        The function that performs (Leapfrog) steps. Takes as arguments (in order)
            step size (containing the direction): float
            inverse mass matrix: Q
            starting point: QP
    potential_energy: Callable[[Q], float]
        The potential energy, of the distribution to be sampled from.
        Takes only the position part (QP.position) as argument
    kinetic_energy: Callable[[Q, Q], float], optional
        Mapping of the momentum to its corresponding kinetic energy. As
        argument the function takes the inverse mass matrix and the momentum.
    max_tree_depth: int
        An upper bound on the 'depth' argument, but has no effect on the functions behaviour.
        It's only required to statically set the size of the `S` array (Q).
    """
    # 1. choose start point of integration
    z = select(go_right, initial_tree.right, initial_tree.left)
    depth = initial_tree.depth
    max_num_proposals = 2**depth
    # 2. build / collect new states
    # Create a storage for left endpoints of subtrees. Size is determined
    # statically by the `max_tree_depth` parameter.
    # NOTE, let's hope this does not break anything but in principle we only
    # need `max_tree_depth` element even though the tree can be of length `max_tree_depth +
    # 1`. This is because we will never access the last element.
    S = tree_util.tree_map(lambda proto: np.empty_like(proto, shape=(max_tree_depth, ) + np.shape(proto)), z)

    z = stepper(np.where(go_right, 1., -1.) * step_size, inverse_mass_matrix, z)
    neg_energy = -total_energy_of_qp(z, potential_energy, partial(kinetic_energy, inverse_mass_matrix))
    diverging = np.abs(neg_energy - initial_neg_energy) > max_energy_difference
    incomplete_tree = Tree(left=z, right=z, logweight=neg_energy, proposal_candidate=z, turning=False, diverging=diverging, depth=-1)
    S = tree_index_update(S, 0, z)

    def amend_incomplete_tree(state):
        n, incomplete_tree, z, S, key = state

        key, key_choose_candidate = random.split(key)
        z = stepper(np.where(go_right, 1., -1.) * step_size, inverse_mass_matrix, z)
        incomplete_tree = add_single_qp_to_tree(key_choose_candidate, incomplete_tree, z, go_right, potential_energy, kinetic_energy, inverse_mass_matrix, initial_neg_energy=initial_neg_energy, max_energy_difference=max_energy_difference)

        def _even_fun(S):
            # n is even, the current z is w.l.o.g. a left endpoint of some
            # subtrees. Register the current z to be used in turning condition
            # checks later, when the right endpoints of it's subtrees are
            # generated.
            S = tree_index_update(S, population_count(n), z)
            return S, False

        def _odd_fun(S):
            # n is odd, the current z is w.l.o.g a right endpoint of some
            # subtrees. Check turning condition against all left endpoints of
            # subtrees that have the current z (/n) as their right endpoint.

            # l = nubmer of subtrees that have current z as their right endpoint.
            l = count_trailing_ones(n)
            # inclusive indices into S referring to the left endpoints of the l subtrees.
            i_max_incl = population_count(n-1)
            i_min_incl = i_max_incl - l + 1
            # TODO: this should traverse the range in reverse
            turning = fori_loop(
                lower = i_min_incl,
                upper = i_max_incl + 1,
                # TODO: conditional for early termination
                body_fun = lambda k, turning: turning | is_euclidean_uturn(tree_index_get(S, k), z),
                init_val = False
            )
            return S, turning

        S, turning = cond(
            pred = n % 2 == 0,
            true_fun = _even_fun,
            false_fun = _odd_fun,
            operand = S
        )
        incomplete_tree = incomplete_tree._replace(turning=turning)
        return (n+1, incomplete_tree, z, S, key)

    def _cont_cond(state):
        n, incomplete_tree, *_ = state
        return (n < max_num_proposals) & (~incomplete_tree.turning) & (~incomplete_tree.diverging)

    n, incomplete_tree, *_ = while_loop(
        # while n < 2**depth and not stop
        cond_fun=_cont_cond,
        body_fun=amend_incomplete_tree,
        init_val=(1, incomplete_tree, z, S, key)
    )

    global _DEBUG_FLAG
    if _DEBUG_FLAG:
        host_callback.call(_DEBUG_FINISH_SUBTREE, None)

    # The depth of a tree which was aborted early is possibly ill defined
    depth = np.where(n == max_num_proposals, depth, -1)
    return incomplete_tree._replace(depth=depth)


def add_single_qp_to_tree(key, tree, qp, go_right, potential_energy, kinetic_energy, inverse_mass_matrix, initial_neg_energy, max_energy_difference):
    """Helper function for progressive sampling. Takes a tree with a sample, and
    a new endpoint, propagates sample.
    """
    # This is technically just a special case of merge_trees with one of the
    # trees being a singleton, depth 0 tree. However, no turning check is
    # required and it is not possible to bias the transition.
    left, right = select(go_right, (tree.left, qp), (qp, tree.right))

    neg_energy = -total_energy_of_qp(qp, potential_energy, partial(kinetic_energy, inverse_mass_matrix))
    diverging = np.abs(neg_energy - initial_neg_energy) > max_energy_difference
    # ln(e^-H_1 + e^-H_2)
    total_logweight = np.logaddexp(tree.logweight, neg_energy)
    # expit(x-y) := 1 / (1 + e^(-(x-y))) = 1 / (1 + e^(y-x)) = e^x / (e^y + e^x)
    prob_of_keeping_old = expit(tree.logweight - neg_energy)
    remain = random.bernoulli(key, prob_of_keeping_old)
    proposal_candidate = select(remain, tree.proposal_candidate, qp)
    # NOTE, set an invalid depth as to indicate that adding a single QP to a
    # perfect binary tree does not yield another perfect binary tree
    return Tree(left, right, total_logweight, proposal_candidate, turning=tree.turning, diverging=diverging, depth=-1)

def merge_trees(key, current_subtree, new_subtree, go_right, bias_transition):
    """Merges two trees, propagating the proposal_candidate"""
    # 5. decide which sample to take based on total weights (merge trees)
    if bias_transition:
        # Bias the transition towards the new subtree (see Betancourt
        # conceptual intro (and Numpyro))
        transition_probability = np.minimum(1., np.exp(new_subtree.logweight - current_subtree.logweight))
    else:
        # expit(x-y) := 1 / (1 + e^(-(x-y))) = 1 / (1 + e^(y-x)) = e^x / (e^y + e^x)
        transition_probability = expit(new_subtree.logweight - current_subtree.logweight)
    # print(f"prob of choosing new sample: {transition_probability}")
    new_sample = select(
        random.bernoulli(key, transition_probability),
        new_subtree.proposal_candidate,
        current_subtree.proposal_candidate
    )
    # 6. define new tree
    left, right = select(
        go_right,
        (current_subtree.left, new_subtree.right),
        (new_subtree.left, current_subtree.right),
    )
    turning = is_euclidean_uturn(left, right)
    diverging = current_subtree.diverging | new_subtree.diverging
    neg_energy = np.logaddexp(new_subtree.logweight, current_subtree.logweight)
    merged_tree = Tree(left=left, right=right, logweight=neg_energy, proposal_candidate=new_sample, turning=turning, diverging=diverging, depth=current_subtree.depth + 1)
    return merged_tree


def count_trailing_ones(n):
    """Count the number of trailing, consecutive ones in the binary
    representation of `n`.

    Warning
    -------
    `n` must be positive and strictly smaller than 2**64

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
        (qp_right.momentum.dot(qp_right.position - qp_left.position) < 0.)
        & (qp_left.momentum.dot(qp_left.position - qp_right.position) < 0.)
    )


class NUTSChain:
    def __init__(self, potential_energy, inverse_mass_matrix, initial_position, key, step_size:float = 1.0, max_tree_depth:int = 10, compile:bool=True, dbg_info:bool=False, bias_transition:bool=True, max_energy_difference:float=np.inf):
        if not callable(potential_energy):
            raise TypeError()
        if not isinstance(step_size, float):
            raise TypeError()
        if not isinstance(max_tree_depth, int):
            raise TypeError()
        if not isinstance(key, np.ndarray):
            if isinstance(key, int):
                key = random.PRNGKey(key)
            else:
                raise TypeError()

        self.last_state = (key, initial_position)
        self.potential_energy = potential_energy

        if isinstance(inverse_mass_matrix, float):
            self.inverse_mass_matrix = tree_util.tree_map(lambda arr: np.full(arr.shape, inverse_mass_matrix), initial_position)
        elif tree_util.tree_structure(inverse_mass_matrix) == tree_util.tree_structure(initial_position):
            shape_match_tree = tree_util.tree_map(lambda a1, a2: a1.shape == a2.shape, inverse_mass_matrix, initial_position)
            shape_and_structure_match = all(tree_util.tree_flatten(shape_match_tree))
            if shape_and_structure_match:
                self.inverse_mass_matrix = inverse_mass_matrix
            else:
                raise ValueError("inverse_mass_matrix has same tree_structe as initial_position but shapes don't match up")
        else:
            te = 'inverse_mass_matrix must either be float or have same tree structure as initial_position'
            raise TypeError(te)
        self.mass_matrix_sqrt = self.inverse_mass_matrix**(-0.5)

        self.step_size = step_size

        def kinetic_energy(inverse_mass_matrix, momentum):
            # NOTE, assume a diagonal mass-matrix
            return inverse_mass_matrix.dot(momentum**2) / 2.

        self.kinetic_energy = kinetic_energy
        kinetic_energy_gradient = lambda inv_m, mom: inv_m * mom
        potential_energy_gradient = grad(self.potential_energy)
        self.stepper = partial(leapfrog_step, potential_energy_gradient, kinetic_energy_gradient)

        self.max_tree_depth = max_tree_depth

        self.max_energy_difference = max_energy_difference
        self.bias_transition = bias_transition

        self.compile = compile
        self.dbg_info = dbg_info

    def generate_n_samples(self, num_samples, _state: Optional[tuple[np.ndarray, Q]] = None) -> Chain:
        _state = self.last_state if _state is None else _state
        key, initial_position = self.last_state

        samples = tree_util.tree_map(lambda arr: np.empty_like(arr, shape=(num_samples,) + np.shape(arr)), initial_position)
        depths = np.empty(num_samples, dtype=np.uint8)
        divergences = np.empty(num_samples, dtype=bool)
        chain = Chain(samples=samples, divergences=divergences, depths=depths)
        if self.dbg_info:
            resampled_momenta = tree_util.tree_map(lambda arr: np.empty_like(initial_position, shape=(num_samples,) + np.shape(arr)), initial_position)
            _qp_proto = QP(initial_position, initial_position)
            _tree_proto = Tree(_qp_proto, _qp_proto, 0., _qp_proto, True, True, 0)
            trees = tree_util.tree_map(
                lambda leaf: np.empty_like(leaf, shape=(num_samples,)+np.shape(leaf)),
                _tree_proto
            )
            chain = chain._replace(resampled_momenta=resampled_momenta, trees=trees)

        def amend_chain(idx, state):
            key, prev_position, chain = state
            key, key_momentum, key_nuts = random.split(key, 3)

            resampled_momentum = sample_momentum_from_diagonal(
                key=key_momentum,
                mass_matrix_sqrt=self.mass_matrix_sqrt
            )
            qp = QP(position=prev_position, momentum=resampled_momentum)

            tree = _generate_nuts_tree(
                initial_qp = qp,
                key = key_nuts,
                step_size = self.step_size,
                max_tree_depth = self.max_tree_depth,
                stepper = self.stepper,
                potential_energy = self.potential_energy,
                kinetic_energy = self.kinetic_energy,
                inverse_mass_matrix=self.inverse_mass_matrix,
                bias_transition=self.bias_transition,
                max_energy_difference=self.max_energy_difference
            )

            samples = tree_index_update(chain.samples, idx, tree.proposal_candidate.position)
            divergences = chain.divergences.at[idx].set(tree.diverging)
            depths = chain.depths.at[idx].set(tree.depth)
            chain = chain._replace(samples=samples, divergences=divergences, depths=depths)
            if self.dbg_info:
                resampled_momenta = tree_index_update(chain.resampled_momenta, idx, resampled_momentum)
                trees = tree_index_update(chain.trees, idx, tree)
                chain = chain._replace(resampled_momenta=resampled_momenta, trees=trees)

            return (key, tree.proposal_candidate.position, chain)

        # TODO: pass initial state as argument and donate to JIT
        chain_assembly = lambda: fori_loop(lower=0, upper=num_samples, body_fun=amend_chain, init_val=(key, initial_position, chain))

        if self.compile:
            final_state = jit(chain_assembly)()
        else:
            final_state = chain_assembly()
        self.last_state = final_state[:2]
        return final_state[2]


class HMCChain:
    def __init__(self, potential_energy, inverse_mass_matrix, initial_position, key, num_steps, step_size: float = 1.0, compile=True, dbg_info=False, max_energy_difference:float=np.inf):
        if not callable(potential_energy):
            raise TypeError()
        if not isinstance(num_steps, int):
            raise TypeError()
        if not isinstance(step_size, float):
            raise TypeError()
        if not isinstance(key, np.ndarray):
            if isinstance(key, int):
                key = random.PRNGKey(key)
            else:
                raise TypeError()

        self.last_state = (key, initial_position)
        self.potential_energy = potential_energy

        if isinstance(inverse_mass_matrix, float):
            self.inverse_mass_matrix = tree_util.tree_map(lambda arr: np.full(arr.shape, inverse_mass_matrix), initial_position)
        elif tree_util.tree_structure(inverse_mass_matrix) == tree_util.tree_structure(initial_position):
            shape_match_tree = tree_util.tree_map(lambda a1, a2: a1.shape == a2.shape, inverse_mass_matrix, initial_position)
            shape_and_structure_match = all(tree_util.tree_flatten(shape_match_tree))
            if shape_and_structure_match:
                self.inverse_mass_matrix = inverse_mass_matrix
            else:
                raise ValueError("inverse_mass_matrix has same tree_structe as initial_position but shapes don't match up")
        else:
            raise ValueError('inverse_mass_matrix must either be float or have same tree structure as initial_position')
        self.mass_matrix_sqrt = self.inverse_mass_matrix**(-0.5)

        self.num_steps = num_steps
        self.step_size = step_size

        def kinetic_energy(inverse_mass_matrix, momentum):
            # NOTE, assume a diagonal mass-matrix
            return inverse_mass_matrix.dot(momentum**2) / 2.

        self.kinetic_energy = kinetic_energy
        kinetic_energy_gradient = lambda inv_m, mom: inv_m * mom
        potential_energy_gradient = grad(self.potential_energy)
        self.stepper = partial(leapfrog_step, potential_energy_gradient, kinetic_energy_gradient)

        self.max_energy_difference = max_energy_difference

        self.compile = compile
        self.dbg_info = dbg_info

    def generate_n_samples(self, num_samples, _state: Optional[tuple[np.ndarray, Q]] = None) -> Chain:
        _state = self.last_state if _state is None else _state
        key, initial_position = self.last_state

        samples = tree_util.tree_map(lambda arr: np.empty_like(arr, shape=(num_samples,) + np.shape(arr)), initial_position)
        divergences = np.empty(num_samples, dtype=bool)
        chain = Chain(samples=samples, divergences=divergences, acceptance=np.array(0.))
        if self.dbg_info:
            resampled_momenta = tree_util.tree_map(lambda arr: np.empty_like(initial_position, shape=(num_samples,) + np.shape(arr)), initial_position)
            _qp_proto = QP(initial_position, initial_position)
            _acc_rej_proto = AcceptedAndRejected(_qp_proto, _qp_proto, True, True)
            trees = tree_util.tree_map(
                lambda leaf: np.empty_like(leaf, shape=(num_samples,)+np.shape(leaf)),
                _acc_rej_proto
            )
            chain = chain._replace(resampled_momenta=resampled_momenta, trees=trees)

        def amend_chain(idx, state):
            key, prev_position, chain = state
            key, key_choose, key_momentum_resample = random.split(key, 3)

            resampled_momentum = sample_momentum_from_diagonal(
                key=key_momentum_resample,
                mass_matrix_sqrt=self.mass_matrix_sqrt
            )
            qp = QP(position=prev_position, momentum=resampled_momentum)

            acc_rej = _generate_hmc_acc_rej(
                key = key_choose,
                initial_qp = qp,
                potential_energy = self.potential_energy,
                kinetic_energy = self.kinetic_energy,
                inverse_mass_matrix=self.inverse_mass_matrix,
                stepper = self.stepper,
                num_steps = self.num_steps,
                step_size = self.step_size,
                max_energy_difference=self.max_energy_difference
            )

            samples = tree_index_update(chain.samples, idx, acc_rej.accepted_qp.position)
            divergences = chain.divergences.at[idx].set(acc_rej.diverging)
            acceptance = (chain.acceptance + (acc_rej.accepted - chain.acceptance) / (idx + 1))
            chain = chain._replace(samples=samples, divergences=divergences, acceptance=acceptance)
            if self.dbg_info:
                resampled_momenta = tree_index_update(chain.resampled_momenta, idx, resampled_momentum)
                trees = tree_index_update(chain.trees, idx, acc_rej)
                chain = chain._replace(resampled_momenta=resampled_momenta, trees=trees)

            return (key, acc_rej.accepted_qp.position, chain)

        # TODO: pass initial state as argument and donate to JIT
        chain_assembly = lambda: fori_loop(lower=0, upper=num_samples, body_fun=amend_chain, init_val=(key, initial_position, chain))

        if self.compile:
            final_state = jit(chain_assembly)()
        else:
            final_state = chain_assembly()
        self.last_state = final_state[:2]
        return final_state[2]
