import numpy as np
from functools import partial
from jax import numpy as jnp
from jax import random, tree_util
from jax import jit, grad

from typing import Callable, NamedTuple, Optional, Union

from .disable_jax_control_flow import fori_loop
from .hmc import Q, QP, Tree, AcceptedAndRejected
from .hmc import (
    generate_hmc_acc_rej, generate_nuts_tree, leapfrog_step,
    sample_momentum_from_diagonal, tree_index_update
)


def _parse_diag_mass_matrix(mass_matrix, x0: Q) -> Q:
    if isinstance(mass_matrix,
                  (float, jnp.ndarray)) and jnp.size(mass_matrix) == 1:
        mass_matrix = tree_util.tree_map(
            partial(jnp.full_like, fill_value=mass_matrix), x0
        )
    elif tree_util.tree_structure(mass_matrix) == tree_util.tree_structure(x0):
        shape_match_tree = tree_util.tree_map(
            lambda a1, a2: jnp.shape(a1) == jnp.shape(a2), mass_matrix, x0
        )
        shape_and_structure_match = all(
            tree_util.tree_flatten(shape_match_tree)
        )
        if not shape_and_structure_match:
            ve = "matrix has same tree_structe as the position but shapes do not match up"
            raise ValueError(ve)
    else:
        te = "matrix must either be float or have same tree structure as the position"
        raise TypeError(te)

    return mass_matrix


class Chain(NamedTuple):
    """Object carrying chain metadata; think: transposed Tree with new axis.
    """
    # Q but with one more dimension on the first axes of the leave tensors
    samples: Q
    divergences: jnp.ndarray
    acceptance: Union[jnp.ndarray, float]
    depths: Optional[jnp.ndarray] = None
    trees: Optional[Union[Tree, AcceptedAndRejected]] = None


class NUTSChain:
    def __init__(
        self,
        potential_energy: Callable[[Q], Union[float, jnp.ndarray]],
        inverse_mass_matrix,
        initial_position: Q,
        key,
        step_size: float = 1.0,
        max_tree_depth: int = 10,
        compile: bool = True,
        dbg_info: bool = False,
        bias_transition: bool = True,
        max_energy_difference: float = jnp.inf
    ):
        if not callable(potential_energy):
            raise TypeError()
        if not isinstance(step_size, float):
            raise TypeError()
        if not isinstance(max_tree_depth, int):
            raise TypeError()
        if not isinstance(key, (jnp.ndarray, np.ndarray)):
            if isinstance(key, int):
                key = random.PRNGKey(key)
            else:
                raise TypeError()

        self.last_state = (key, initial_position)
        self.potential_energy = potential_energy

        self.inverse_mass_matrix = _parse_diag_mass_matrix(
            inverse_mass_matrix, x0=initial_position
        )
        self.mass_matrix_sqrt = self.inverse_mass_matrix**(-0.5)

        self.step_size = step_size

        def kinetic_energy(inverse_mass_matrix, momentum):
            # NOTE, assume a diagonal mass-matrix
            return inverse_mass_matrix.dot(momentum**2) / 2.

        self.kinetic_energy = kinetic_energy
        kinetic_energy_gradient = lambda inv_m, mom: inv_m * mom
        potential_energy_gradient = grad(self.potential_energy)
        self.stepper = partial(
            leapfrog_step, potential_energy_gradient, kinetic_energy_gradient
        )

        self.max_tree_depth = max_tree_depth

        self.max_energy_difference = max_energy_difference
        self.bias_transition = bias_transition

        self.compile = compile
        self.dbg_info = dbg_info

        def sample_next_state(key, prev_position: Q):
            key, key_momentum, key_nuts = random.split(key, 3)

            resampled_momentum = sample_momentum_from_diagonal(
                key=key_momentum, mass_matrix_sqrt=self.mass_matrix_sqrt
            )
            qp = QP(position=prev_position, momentum=resampled_momentum)

            tree = generate_nuts_tree(
                initial_qp=qp,
                key=key_nuts,
                step_size=self.step_size,
                max_tree_depth=self.max_tree_depth,
                stepper=self.stepper,
                potential_energy=self.potential_energy,
                kinetic_energy=self.kinetic_energy,
                inverse_mass_matrix=self.inverse_mass_matrix,
                bias_transition=self.bias_transition,
                max_energy_difference=self.max_energy_difference
            )
            return tree, (key, tree.proposal_candidate.position)

        self.sample_next_state = sample_next_state

        def update_chain(chain, idx, tree):
            num_proposals = 2**tree.depth - 1
            tree_acceptance = jnp.where(
                num_proposals > 0, tree.cumulative_acceptance / num_proposals,
                0.
            )

            samples = tree_index_update(
                chain.samples, idx, tree.proposal_candidate.position
            )
            divergences = chain.divergences.at[idx].set(tree.diverging)
            depths = chain.depths.at[idx].set(tree.depth)
            acceptance = (
                chain.acceptance + (tree_acceptance - chain.acceptance) /
                (idx + 1)
            )
            chain = chain._replace(
                samples=samples,
                divergences=divergences,
                acceptance=acceptance,
                depths=depths
            )
            if self.dbg_info:
                trees = tree_index_update(chain.trees, idx, tree)
                chain = chain._replace(trees=trees)

            return chain

        self.update_chain = update_chain

    def generate_n_samples(
        self,
        num_samples,
        _state: Optional[tuple[jnp.ndarray, Q]] = None
    ) -> Chain:
        _state = self.last_state if _state is None else _state
        key, initial_position = self.last_state

        samples = tree_util.tree_map(
            lambda arr: jnp.
            empty_like(arr, shape=(num_samples, ) + jnp.shape(arr)),
            initial_position
        )
        depths = jnp.empty(num_samples, dtype=jnp.uint8)
        divergences = jnp.empty(num_samples, dtype=bool)
        chain = Chain(
            samples=samples,
            divergences=divergences,
            acceptance=0.,
            depths=depths
        )
        if self.dbg_info:
            _qp_proto = QP(initial_position, initial_position)
            _tree_proto = Tree(
                _qp_proto,
                _qp_proto,
                0.,
                _qp_proto,
                turning=True,
                diverging=True,
                depth=0,
                cumulative_acceptance=0.
            )
            trees = tree_util.tree_map(
                lambda leaf: jnp.
                empty_like(leaf, shape=(num_samples, ) + jnp.shape(leaf)),
                _tree_proto
            )
            chain = chain._replace(trees=trees)

        def amend_chain(idx, state):
            core_state, chain = state
            tree, core_state = self.sample_next_state(*core_state)
            chain = self.update_chain(chain, idx, tree)
            return core_state, chain

        # TODO: pass initial state as argument and donate to JIT
        chain_assembly = lambda: fori_loop(
            lower=0,
            upper=num_samples,
            body_fun=amend_chain,
            init_val=((key, initial_position), chain)
        )

        if self.compile:
            final_state = jit(chain_assembly)()
        else:
            final_state = chain_assembly()
        self.last_state, chain = final_state
        return chain


class HMCChain:
    def __init__(
        self,
        potential_energy: Callable,
        inverse_mass_matrix,
        initial_position,
        key,
        num_steps,
        step_size: float = 1.0,
        compile=True,
        dbg_info=False,
        max_energy_difference: float = jnp.inf
    ):
        if not callable(potential_energy):
            raise TypeError()
        if not isinstance(num_steps, int):
            raise TypeError()
        if not isinstance(step_size, float):
            raise TypeError()
        if not isinstance(key, (jnp.ndarray, np.ndarray)):
            if isinstance(key, int):
                key = random.PRNGKey(key)
            else:
                raise TypeError()

        self.last_state = (key, initial_position)
        self.potential_energy = potential_energy

        self.inverse_mass_matrix = _parse_diag_mass_matrix(
            inverse_mass_matrix, x0=initial_position
        )
        self.mass_matrix_sqrt = self.inverse_mass_matrix**(-0.5)

        self.num_steps = num_steps
        self.step_size = step_size

        def kinetic_energy(inverse_mass_matrix, momentum):
            # NOTE, assume a diagonal mass-matrix
            return inverse_mass_matrix.dot(momentum**2) / 2.

        self.kinetic_energy = kinetic_energy
        kinetic_energy_gradient = lambda inv_m, mom: inv_m * mom
        potential_energy_gradient = grad(self.potential_energy)
        self.stepper = partial(
            leapfrog_step, potential_energy_gradient, kinetic_energy_gradient
        )

        self.max_energy_difference = max_energy_difference

        self.compile = compile
        self.dbg_info = dbg_info

        def sample_next_state(key, prev_position):
            key, key_choose, key_momentum_resample = random.split(key, 3)

            resampled_momentum = sample_momentum_from_diagonal(
                key=key_momentum_resample,
                mass_matrix_sqrt=self.mass_matrix_sqrt
            )
            qp = QP(position=prev_position, momentum=resampled_momentum)

            acc_rej = generate_hmc_acc_rej(
                key=key_choose,
                initial_qp=qp,
                potential_energy=self.potential_energy,
                kinetic_energy=self.kinetic_energy,
                inverse_mass_matrix=self.inverse_mass_matrix,
                stepper=self.stepper,
                num_steps=self.num_steps,
                step_size=self.step_size,
                max_energy_difference=self.max_energy_difference
            )
            return acc_rej, (key, acc_rej.accepted_qp.position)

        self.sample_next_state = sample_next_state

        def update_chain(chain, idx, acc_rej):
            samples = tree_index_update(
                chain.samples, idx, acc_rej.accepted_qp.position
            )
            divergences = chain.divergences.at[idx].set(acc_rej.diverging)
            acceptance = (
                chain.acceptance + (acc_rej.accepted - chain.acceptance) /
                (idx + 1)
            )
            chain = chain._replace(
                samples=samples, divergences=divergences, acceptance=acceptance
            )
            if self.dbg_info:
                trees = tree_index_update(chain.trees, idx, acc_rej)
                chain = chain._replace(trees=trees)
            return chain

        self.update_chain = update_chain

    # TODO: merge NUTSChain and HMCChain into a joined method
    def generate_n_samples(
        self,
        num_samples,
        _state: Optional[tuple[jnp.ndarray, Q]] = None
    ) -> Chain:
        _state = self.last_state if _state is None else _state
        key, initial_position = self.last_state

        samples = tree_util.tree_map(
            lambda arr: jnp.
            empty_like(arr, shape=(num_samples, ) + jnp.shape(arr)),
            initial_position
        )
        divergences = jnp.empty(num_samples, dtype=bool)
        chain = Chain(samples=samples, divergences=divergences, acceptance=0.)
        if self.dbg_info:
            _qp_proto = QP(initial_position, initial_position)
            _acc_rej_proto = AcceptedAndRejected(
                _qp_proto, _qp_proto, True, True
            )
            trees = tree_util.tree_map(
                lambda leaf: jnp.
                empty_like(leaf, shape=(num_samples, ) + jnp.shape(leaf)),
                _acc_rej_proto
            )
            chain = chain._replace(trees=trees)

        def amend_chain(idx, state):
            core_state, chain = state
            acc_rej, core_state = self.sample_next_state(*core_state)
            chain = self.update_chain(chain, idx, acc_rej)
            return core_state, chain

        # TODO: pass initial state as argument and donate to JIT
        chain_assembly = lambda: fori_loop(
            lower=0,
            upper=num_samples,
            body_fun=amend_chain,
            init_val=((key, initial_position), chain)
        )

        if self.compile:
            final_state = jit(chain_assembly)()
        else:
            final_state = chain_assembly()
        self.last_state, chain = final_state
        return chain
