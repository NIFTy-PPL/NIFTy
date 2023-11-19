# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import numpy as np
from jax import grad
from jax import numpy as jnp
from jax import random, tree_util

from .lax import fori_loop
from .hmc import AcceptedAndRejected, Q, QP, Tree
from .hmc import (
    generate_hmc_acc_rej,
    generate_nuts_tree,
    leapfrog_step,
    sample_momentum_from_diagonal,
    tree_index_update,
)


def _parse_diag_mass_matrix(mass_matrix, position_proto: Q) -> Q:
    if isinstance(mass_matrix,
                  (float, jnp.ndarray)) and jnp.size(mass_matrix) == 1:
        mass_matrix = tree_util.tree_map(
            partial(jnp.full_like, fill_value=mass_matrix), position_proto
        )
    elif tree_util.tree_structure(mass_matrix
                                 ) == tree_util.tree_structure(position_proto):
        shape_match_tree = tree_util.tree_map(
            lambda a1, a2: jnp.shape(a1) == jnp.shape(a2), mass_matrix,
            position_proto
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


class _Sampler:
    def __init__(
        self,
        potential_energy: Callable[[Q], Union[jnp.ndarray, float]],
        inverse_mass_matrix,
        position_proto: Q,
        step_size: Union[jnp.ndarray, float] = 1.0,
        max_energy_difference: Union[jnp.ndarray, float] = jnp.inf
    ):
        if not callable(potential_energy):
            raise TypeError()
        if not isinstance(step_size, (jnp.ndarray, float)):
            raise TypeError()

        self.potential_energy = potential_energy

        self.inverse_mass_matrix = _parse_diag_mass_matrix(
            inverse_mass_matrix, position_proto=position_proto
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

        self.max_energy_difference = max_energy_difference

        def sample_next_state(key,
                              prev_position: Q) -> Tuple[Any, Tuple[Any, Q]]:
            raise NotImplementedError()

        self.sample_next_state = sample_next_state

    @staticmethod
    def init_chain(
        num_samples: int, position_proto, save_intermediates: bool
    ) -> Chain:
        raise NotImplementedError()

    @staticmethod
    def update_chain(
        chain: Chain, idx: Union[jnp.ndarray, int], tree: Tree
    ) -> Chain:
        raise NotImplementedError()

    def generate_n_samples(
        self,
        key: Any,
        initial_position: Q,
        num_samples,
        *,
        save_intermediates: bool = False
    ) -> Tuple[Chain, Tuple[Any, Q]]:
        if not isinstance(key, (jnp.ndarray, np.ndarray)):
            if isinstance(key, int):
                key = random.PRNGKey(key)
            else:
                raise TypeError()

        chain = self.init_chain(
            num_samples, initial_position, save_intermediates
        )

        def amend_chain(idx, state):
            chain, core_state = state
            tree, core_state = self.sample_next_state(*core_state)
            chain = self.update_chain(chain, idx, tree)
            return chain, core_state

        chain, core_state = fori_loop(
            lower=0,
            upper=num_samples,
            body_fun=amend_chain,
            init_val=(chain, (key, initial_position))
        )

        return chain, core_state


class NUTSChain(_Sampler):
    def __init__(
        self,
        potential_energy: Callable[[Q], Union[float, jnp.ndarray]],
        inverse_mass_matrix,
        position_proto: Q,
        step_size: float = 1.0,
        max_tree_depth: int = 10,
        bias_transition: bool = True,
        max_energy_difference: float = jnp.inf
    ):
        super().__init__(
            potential_energy=potential_energy,
            inverse_mass_matrix=inverse_mass_matrix,
            position_proto=position_proto,
            step_size=step_size,
            max_energy_difference=max_energy_difference
        )

        if not isinstance(max_tree_depth, int):
            raise TypeError()
        self.bias_transition = bias_transition
        self.max_tree_depth = max_tree_depth

        def sample_next_state(key,
                              prev_position: Q) -> Tuple[Tree, Tuple[Any, Q]]:
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

    @staticmethod
    def init_chain(
        num_samples: int, position_proto, save_intermediates: bool
    ) -> Chain:
        samples = tree_util.tree_map(
            lambda arr: jnp.
            zeros_like(arr, shape=(num_samples, ) + jnp.shape(arr)),
            position_proto
        )
        depths = jnp.zeros(num_samples, dtype=jnp.uint64)
        divergences = jnp.zeros(num_samples, dtype=bool)
        chain = Chain(
            samples=samples,
            divergences=divergences,
            acceptance=0.,
            depths=depths
        )
        if save_intermediates:
            _qp_proto = QP(position_proto, position_proto)
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
                zeros_like(leaf, shape=(num_samples, ) + jnp.shape(leaf)),
                _tree_proto
            )
            chain = chain._replace(trees=trees)

        return chain

    @staticmethod
    def update_chain(
        chain: Chain, idx: Union[jnp.ndarray, int], tree: Tree
    ) -> Chain:
        num_proposals = 2**jnp.array(tree.depth, dtype=jnp.uint64) - 1
        tree_acceptance = jnp.where(
            num_proposals > 0, tree.cumulative_acceptance / num_proposals, 0.
        )

        samples = tree_index_update(
            chain.samples, idx, tree.proposal_candidate.position
        )
        divergences = chain.divergences.at[idx].set(tree.diverging)
        depths = chain.depths.at[idx].set(tree.depth)
        acceptance = (
            chain.acceptance + (tree_acceptance - chain.acceptance) / (idx + 1)
        )
        chain = chain._replace(
            samples=samples,
            divergences=divergences,
            acceptance=acceptance,
            depths=depths
        )
        if chain.trees is not None:
            trees = tree_index_update(chain.trees, idx, tree)
            chain = chain._replace(trees=trees)

        return chain


class HMCChain(_Sampler):
    def __init__(
        self,
        potential_energy: Callable,
        inverse_mass_matrix,
        position_proto,
        num_steps,
        step_size: float = 1.0,
        max_energy_difference: float = jnp.inf
    ):
        super().__init__(
            potential_energy=potential_energy,
            inverse_mass_matrix=inverse_mass_matrix,
            position_proto=position_proto,
            step_size=step_size,
            max_energy_difference=max_energy_difference
        )

        if not isinstance(num_steps, (jnp.ndarray, int)):
            raise TypeError()
        self.num_steps = num_steps

        def sample_next_state(key,
                              prev_position: Q) -> Tuple[Tree, Tuple[Any, Q]]:
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

    @staticmethod
    def init_chain(
        num_samples: int, position_proto, save_intermediates: bool
    ) -> Chain:
        samples = tree_util.tree_map(
            lambda arr: jnp.
            zeros_like(arr, shape=(num_samples, ) + jnp.shape(arr)),
            position_proto
        )
        divergences = jnp.zeros(num_samples, dtype=bool)
        chain = Chain(samples=samples, divergences=divergences, acceptance=0.)
        if save_intermediates:
            _qp_proto = QP(position_proto, position_proto)
            _acc_rej_proto = AcceptedAndRejected(
                _qp_proto, _qp_proto, True, True
            )
            trees = tree_util.tree_map(
                lambda leaf: jnp.
                zeros_like(leaf, shape=(num_samples, ) + jnp.shape(leaf)),
                _acc_rej_proto
            )
            chain = chain._replace(trees=trees)

        return chain

    @staticmethod
    def update_chain(
        chain: Chain, idx: Union[jnp.ndarray, int], acc_rej: AcceptedAndRejected
    ) -> Chain:
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
        if chain.trees is not None:
            trees = tree_index_update(chain.trees, idx, acc_rej)
            chain = chain._replace(trees=trees)

        return chain
