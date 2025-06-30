#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Philipp Frank, Jakob Roth, Gordian Edenhofer

import inspect
import os
import pickle
from dataclasses import field
from functools import partial
from os import makedirs
from typing import Any, Callable, Literal, NamedTuple, Optional, TypeVar, Union

import jax
import numpy as np
from jax import numpy as jnp
from jax import random
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as Pspec
from jax.tree_util import Partial, tree_map

from . import optimize
from .evi import (
    Samples,
    _parse_jit,
    concatenate_zip,
    draw_linear_residual,
    nonlinearly_update_residual,
)
from .likelihood import Likelihood
from .logger import logger
from .minisanity import minisanity
from .model import LazyModel
from .tree_math import get_map, hide_strings, vdot

P = TypeVar("P")


def get_status_message(samples, state, residual=None, *, name="", map="lmap") -> str:
    energy = state.minimization_state.fun
    msg_smpl = ""
    if isinstance(state.sample_state, optimize.OptimizeResults):
        nlsi = tuple(int(el) for el in state.sample_state.nit)
        msg_smpl = f"\n{name}: #(Nonlinear sampling steps) {nlsi}"
    elif isinstance(state.sample_state, (np.ndarray, jax.Array)):
        nlsi = tuple(int(el) for el in state.sample_state)
        msg_smpl = f"\n{name}: Linear sampling status {nlsi}"
    mini_res = ""
    if residual is not None:
        _, mini_res = minisanity(samples, residual, map=map)
    _, mini_pr = minisanity(samples, map=map)
    msg = (
        f"{name}: Iteration {state.nit:04d} ⛰:{energy:+2.4e}"
        f"{msg_smpl}"
        f"\n{name}: #(KL minimization steps) {state.minimization_state.nit}"
        f"\n{name}: Likelihood residual(s):\n{mini_res}"
        f"\n{name}: Prior residual(s):\n{mini_pr}"
        f"\n"
    )
    return msg


_reduce = partial(tree_map, partial(jnp.mean, axis=0))


class _StandardHamiltonian(LazyModel):
    """Joined object storage composed of a user-defined likelihood and a
    standard normal prior.
    """

    likelihood: Likelihood = field(metadata=dict(static=False))

    def __init__(self, likelihood: Likelihood, /):
        self.likelihood = likelihood

    def __call__(self, primals, **primals_kw):
        return self.energy(primals, **primals_kw)

    def energy(self, primals, **primals_kw):
        return self.likelihood(primals, **primals_kw) + 0.5 * vdot(primals, primals)

    def metric(self, primals, tangents, **primals_kw):
        return self.likelihood.metric(primals, tangents, **primals_kw) + tangents


def _kl_vg(
    likelihood,
    primals,
    primals_samples,
    *,
    map=jax.vmap,
    reduce=_reduce,
    named_sharding=None,
    kl_device_map="shard_map",
):
    assert isinstance(primals_samples, Samples)
    map = get_map(map)
    ham = _StandardHamiltonian(likelihood)

    if len(primals_samples) == 0:
        return jax.value_and_grad(ham)(primals)

    if named_sharding is None:
        vvg = map(jax.value_and_grad(ham))
    else:
        if kl_device_map == "shard_map":
            vvg = map(jax.value_and_grad(ham))
            spec_tree = tree_map(lambda x: named_sharding.spec, primals)
            out_spec = (named_sharding.spec, spec_tree)
            in_spec = (spec_tree,)
            vvg = shard_map(
                vvg, mesh=named_sharding.mesh, in_specs=in_spec, out_specs=out_spec
            )
        elif kl_device_map == "jit":
            vvg = map(jax.value_and_grad(ham))
            sharding_tree = tree_map(lambda x: named_sharding, primals)
            out_sharding = (named_sharding, sharding_tree)
            in_sharding = (sharding_tree,)
            vvg = jax.jit(vvg, in_shardings=in_sharding, out_shardings=out_sharding)
        elif kl_device_map == "pmap":
            vvg = jax.pmap(jax.value_and_grad(ham))
        else:
            ve = f"`kl_device_map` need to be `pmap`, `shard_map`, or `jit`, not {kl_device_map}"
            raise ValueError(ve)

    s = vvg(primals_samples.at(primals).samples)
    return reduce(s)


def _kl_met(
    likelihood,
    primals,
    tangents,
    primals_samples,
    *,
    map=jax.vmap,
    reduce=_reduce,
    named_sharding=None,
    kl_device_map="shard_map",
):
    assert isinstance(primals_samples, Samples)
    map = get_map(map)
    ham = _StandardHamiltonian(likelihood)

    if len(primals_samples) == 0:
        return ham.metric(primals, tangents)
    met = Partial(ham.metric, tangents=tangents)

    if named_sharding is None:
        vmet = map(met)
    else:
        if kl_device_map == "shard_map":
            vmet = map(met)
            spec_tree = tree_map(lambda x: named_sharding.spec, primals)
            out_spec = spec_tree
            in_spec = (spec_tree,)
            vmet = shard_map(
                vmet,
                mesh=named_sharding.mesh,
                in_specs=in_spec,
                out_specs=out_spec,
            )
        elif kl_device_map == "jit":
            vmet = map(met)
            sharding_tree = tree_map(lambda x: named_sharding, primals)
            out_sharding = sharding_tree
            in_sharding = (sharding_tree,)
            vmet = jax.jit(vmet, in_shardings=in_sharding, out_shardings=out_sharding)
        elif kl_device_map == "pmap":
            vmet = jax.pmap(met)
        else:
            ve = f"`kl_device_map` need to be `pmap`, `shard_map`, or `jit`, not {kl_device_map}"
            raise ValueError(ve)
    s = vmet(primals_samples.at(primals).samples)
    return reduce(s)


SMPL_MODE_TYP = Literal[
    "linear_sample",
    "linear_resample",
    "nonlinear_sample",
    "nonlinear_resample",
    "nonlinear_update",
]
SMPL_MODE_GENERIC_TYP = Union[SMPL_MODE_TYP, Callable[[int], SMPL_MODE_TYP]]
DICT_OR_CALL4DICT_TYP = Union[Callable[[int], dict], dict]


class OptimizeVIState(NamedTuple):
    nit: int
    key: Any
    sample_state: Optional[optimize.OptimizeResults] = None
    minimization_state: Optional[optimize.OptimizeResults] = None
    config: dict[str, Union[dict, Callable[[int], Any], Any]] = {}


def _getitem_at_nit(config, key, nit):
    c = config[key]
    if callable(c) and len(inspect.getfullargspec(c).args) == 1:
        return c(nit)
    return c


class OptimizeVI:
    """State-less assembly of all methods needed for an MGVI/geoVI style VI
    approximation.

    Builds functions for a VI approximation via variants of the `Geometric
    Variational Inference` and/or `Metric Gaussian Variational Inference`
    algorithms. They produce approximate posterior samples that are used for KL
    estimation internally and the final set of samples are the approximation of
    the posterior. The samples can be linear, i.e. following a standard normal
    distribution in model space, or nonlinear, i.e. following a standard normal
    distribution in the canonical coordinate system of the Riemannian manifold
    associated with the metric of the approximate posterior distribution. The
    coordinate transformation for the nonlinear sample is approximated by an
    expansion.

    Both linear and nonlinear sample start by drawing a sample from the
    inverse metric. To do so, we draw a sample which has the metric as
    covariance structure and apply the inverse metric to it. The sample
    transformed in this way has the inverse metric as covariance. The first
    part is trivial since we can use the left square root of the metric
    :math:`L` associated with every likelihood:

    .. math::
        \\tilde{d} \\leftarrow \\mathcal{G}(0,\\mathbb{1}) \\
        t = L \\tilde{d}

    with :math:`t` now having a covariance structure of

    .. math::
        <t t^\\dagger> = L <\\tilde{d} \\tilde{d}^\\dagger> L^\\dagger = M .

    To transform the sample to an inverse sample, we apply the inverse
    metric. We can do so using the conjugate gradient algorithm (CG). The CG
    algorithm yields the solution to :math:`M s = t`, i.e. applies the
    inverse of :math:`M` to :math:`t`:

    .. math::
        M &s =  t \\\\
        &s = M^{-1} t = cg(M, t) .

    The linear sample is :math:`s`.

    The nonlinear sampling uses :math:`s` as a starting value and curves it in
    a nonlinear way as to better resemble the posterior locally. See the below
    reference literature for more details on the nonlinear sampling.

    See also
    --------
    `Geometric Variational Inference`, Philipp Frank, Reimar Leike,
    Torsten A. Enßlin, `<https://arxiv.org/abs/2105.10470>`_
    `<https://doi.org/10.3390/e23070853>`_
    `Metric Gaussian Variational Inference`, Jakob Knollmüller,
    Torsten A. Enßlin, `<https://arxiv.org/abs/1901.11033>`_
    """

    def __init__(
        self,
        likelihood: Likelihood,
        n_total_iterations: int,
        *,
        kl_jit=True,
        residual_jit=True,
        kl_map=jax.vmap,
        residual_map="lmap",
        kl_reduce=_reduce,
        mirror_samples=True,
        devices: Optional[list] = None,
        kl_device_map="shard_map",
        residual_device_map="pmap",
        _kl_value_and_grad: Optional[Callable] = None,
        _kl_metric: Optional[Callable] = None,
        _draw_linear_residual: Optional[Callable] = None,
        _nonlinearly_update_residual: Optional[Callable] = None,
        _get_status_message: Optional[Callable] = None,
    ):
        """JaxOpt style minimizer for a VI approximation of a distribution with
        samples.

        Parameters
        ----------
        likelihood: :class:`~nifty.re.likelihood.Likelihood`
            Likelihood to be used for inference.
        n_total_iterations: int
            Total number of iterations. One iteration consists of the steps
            1) - 3).
        kl_jit: bool or callable
            Whether to jit the KL minimization.
        residual_jit: bool or callable
            Whether to jit the residual sampling functions.
        kl_map: callable or str
            Map function used for the KL minimization.
        residual_map: callable or str
            Map function used for the residual sampling functions.
        kl_reduce: callable
            Reduce function used for the KL minimization.
        mirror_samples: bool
            Whether to mirror the samples or not.
        devices : list of devices or None
            Devices over which the samples are mapped. If `None` only the
            default device is used. To use all detected devices pass
            jax.devices(). Generally the samples needs to be evenly
            distributable over the devices. For details see the descriptions of
            the `kl_device_map` and `residual_device_map` arguments.
        kl_device_map : str
            Map function used for mapping KL minimization over the devices
            listed in `devices`. `kl_device_map` can be 'shard_map', 'pmap', or
            'jit'. If set to 'pmap', 2*n_samples need to be equal to the number
            of devices. For all other maps the samples needs to be equally
            distributable over the devices.
        residual_device_map : str
            Map function used for mapping sampling over the devices listed in
            `devices`. `residual_device_map` can be 'shard_map', 'pmap', or
            'jit'. If set to 'pmap', 2*n_samples need to be equal to the number
            of devices. If only linear samples are drawn ,'pmap' also works if
            n_samples equals the number of devices. For the other maps it is
            sufficient if 2*n_samples equals the number of devices, or
            n_samples can be evenly divided by the number of devices.

        Notes
        -----
        Implements the base logic present in conditional VI approximations
        such as MGVI and geoVI. First samples are generated (and/or updated)
        and then their collective mean is optimized for using the sample
        estimated variational KL between the true distribution and the sampled
        approximation. This is split into three steps:
        1) Sample generation
        2) Sample update
        3) KL minimization.
        Step 1) and 2) may be skipped depending on the minimizers state, but
        step 3) is always performed at the end of one iteration. A full loop
        consists of repeatedly iterating over the steps 1) - 3).
        """
        kl_jit = _parse_jit(kl_jit)
        residual_jit = _parse_jit(residual_jit)
        residual_map = get_map(residual_map)
        self.named_sharding = None
        if (not devices is None) and len(devices) > 1:
            mesh = Mesh(devices, ("x",))
            pspec = Pspec("x")
            self.named_sharding = NamedSharding(mesh, pspec)
        self.residual_device_map = residual_device_map

        if mirror_samples is False:
            raise NotImplementedError()

        if _kl_value_and_grad is None:
            _kl_value_and_grad = partial(
                kl_jit(
                    _kl_vg,
                    static_argnames=(
                        "map",
                        "reduce",
                        "named_sharding",
                        "kl_device_map",
                    ),
                ),
                likelihood,
                map=kl_map,
                reduce=kl_reduce,
                named_sharding=self.named_sharding,
                kl_device_map=kl_device_map,
            )
        if _kl_metric is None:
            _kl_metric = partial(
                kl_jit(
                    _kl_met,
                    static_argnames=(
                        "map",
                        "reduce",
                        "named_sharding",
                        "kl_device_map",
                    ),
                ),
                likelihood,
                map=kl_map,
                reduce=kl_reduce,
                named_sharding=self.named_sharding,
                kl_device_map=kl_device_map,
            )
        if _draw_linear_residual is None:
            _draw_linear_residual = partial(
                residual_jit(draw_linear_residual), likelihood
            )
        if _nonlinearly_update_residual is None:
            _nonlinearly_update_residual = partial(
                residual_jit(nonlinearly_update_residual), likelihood
            )
        if _get_status_message is None:
            _get_status_message = partial(
                get_status_message,
                residual=likelihood.normalized_residual,
                name=self.__class__.__name__,
            )

        self.n_total_iterations = n_total_iterations
        self.kl_value_and_grad = _kl_value_and_grad
        self.kl_metric = _kl_metric
        self.draw_linear_residual = _draw_linear_residual
        self.nonlinearly_update_residual = _nonlinearly_update_residual
        self.residual_map = residual_map
        self.get_status_message = _get_status_message

    def draw_linear_samples(self, primals, keys, **kwargs):
        # NOTE, use `Partial` in favor of `partial` to allow the (potentially)
        # re-jitting `residual_map` to trace the kwargs
        kwargs = hide_strings(kwargs)
        if self.named_sharding is None:
            sampler = Partial(self.draw_linear_residual, **kwargs)
            sampler = self.residual_map(sampler, in_axes=(None, 0))
            smpls, smpls_states = sampler(primals, keys)
            # zip samples such that the mirrored-counterpart always comes right
            # after the original sample
            smpls = concatenate_zip(smpls, -smpls)
        else:
            n_samples = len(keys)
            if n_samples == self.named_sharding.mesh.size / 2:
                keys = jnp.repeat(keys, 2, axis=0)
            keys = jax.device_put(keys, self.named_sharding)

            # zip samples such that the mirrored-counterpart always comes right
            # after the original sample. out_shardings is need for telling JAX
            # not to move all samples to a single device.
            @partial(jax.jit, out_shardings=self.named_sharding)
            def concatenate_zip_pmap(*arrays):
                return tree_map(
                    lambda *x: jnp.stack(x, axis=1).reshape((-1,) + x[0].shape[1:]),
                    *arrays,
                )

            @partial(jax.jit, out_shardings=self.named_sharding)
            def _special_mirror_samples(samples):
                return samples.at[1::2].set(-samples[1::2])

            if self.residual_device_map == "pmap":
                sampler = Partial(self.draw_linear_residual, **kwargs)
                sampler = jax.pmap(sampler, in_axes=(None, 0))
                keys = jax.device_put(keys, self.named_sharding)
                smpls, smpls_states = sampler(primals, keys)
            elif self.residual_device_map == "jit":
                sampler = Partial(self.draw_linear_residual, primals, **kwargs)
                sampler = self.residual_map(sampler)
                sampler = jax.jit(
                    sampler,
                    in_shardings=self.named_sharding,
                    out_shardings=self.named_sharding,
                )
                smpls, smpls_states = sampler(keys)
            elif self.residual_device_map == "shard_map":
                sampler = Partial(self.draw_linear_residual, primals, **kwargs)
                out_spec = (
                    tree_map(lambda x: self.named_sharding.spec, primals),
                    self.named_sharding.spec,
                )
                sampler = shard_map(
                    self.residual_map(sampler),
                    mesh=self.named_sharding.mesh,
                    in_specs=self.named_sharding.spec,
                    out_specs=out_spec,
                    check_rep=False,  # FIXME Maybe enable in future JAX releases
                )
                smpls, smpls_states = sampler(keys)
            else:
                ve = f"`residual_device_map` need to be `pmap`, `shard_map`, or `jit`, not {self.residual_device_map}"
                raise ValueError(ve)
            if n_samples == self.named_sharding.mesh.size / 2:
                smpls = tree_map(_special_mirror_samples, smpls)
                keys = keys[::2]  # undo jnp.repeat
            else:
                smpls = concatenate_zip_pmap(smpls, -smpls)

        smpls = Samples(pos=primals, samples=smpls, keys=keys)
        return smpls, smpls_states

    def nonlinearly_update_samples(self, samples: Samples, **kwargs):
        # NOTE, use `Partial` in favor of `partial` to allow the (potentially)
        # re-jitting `residual_map` to trace the kwargs
        kwargs = hide_strings(kwargs)
        assert len(samples.keys) == len(samples) // 2
        metric_sample_key = concatenate_zip(*((samples.keys,) * 2))
        sgn = jnp.ones(len(samples.keys))
        sgn = concatenate_zip(sgn, -sgn)
        if self.named_sharding is None:
            curver = Partial(self.nonlinearly_update_residual, **kwargs)
            curver = self.residual_map(curver, in_axes=(None, 0, 0, 0))
            smpls, smpls_states = curver(
                samples.pos, samples._samples, metric_sample_key, sgn
            )
        else:
            curver = Partial(self.nonlinearly_update_residual, samples.pos, **kwargs)
            if self.residual_device_map == "shard_map":
                spec_tree = tree_map(lambda x: self.named_sharding.spec, samples.pos)
                out_spec = (spec_tree, self.named_sharding.spec)
                in_spec = (
                    spec_tree,
                    self.named_sharding.spec,
                    self.named_sharding.spec,
                )
                curver = shard_map(
                    self.residual_map(curver, in_axes=(0, 0, 0)),
                    mesh=self.named_sharding.mesh,
                    in_specs=in_spec,
                    out_specs=out_spec,
                    check_rep=False,
                )
            elif self.residual_device_map == "jit":
                sharding_tree = tree_map(lambda x: self.named_sharding, samples.pos)
                out_sharding = (sharding_tree, self.named_sharding)
                in_sharding = (sharding_tree, self.named_sharding, self.named_sharding)
                curver = self.residual_map(curver)
                curver = jax.jit(
                    curver, in_shardings=in_sharding, out_shardings=out_sharding
                )
                metric_sample_key = jax.device_put(
                    metric_sample_key, self.named_sharding
                )
            elif self.residual_device_map == "pmap":
                curver = jax.pmap(curver, in_axes=(0, 0, 0))
            else:
                ve = f"`residual_device_map` need to be `pmap`, `shard_map`, or `jit`, not {self.residual_device_map}"
                raise ValueError(ve)
            smpls, smpls_states = curver(samples._samples, metric_sample_key, sgn)

        smpls = Samples(pos=samples.pos, samples=smpls, keys=samples.keys)
        return smpls, smpls_states

    def draw_samples(
        self,
        samples: Samples,
        *,
        key,
        sample_mode: SMPL_MODE_TYP,
        n_samples: int,
        point_estimates,
        draw_linear_kwargs={},
        nonlinearly_update_kwargs={},
        **kwargs,
    ):
        # Always resample if `n_samples` increased
        n_keys = 0 if samples.keys is None else len(samples.keys)
        if n_samples == 0:
            sample_mode = ""
        elif n_samples != n_keys and sample_mode.lower() == "nonlinear_update":
            sample_mode = "nonlinear_resample"
        elif n_samples != n_keys and sample_mode.lower().endswith("_sample"):
            sample_mode = sample_mode.replace("_sample", "_resample")

        if sample_mode.lower() in (
            "linear_resample",
            "linear_sample",
            "nonlinear_resample",
            "nonlinear_sample",
        ):
            k_smpls = samples.keys  # Re-use the keys if not re-sampling
            if sample_mode.lower().endswith("_resample"):
                k_smpls = random.split(key, n_samples)
            assert n_samples == len(k_smpls)
            samples, st_smpls = self.draw_linear_samples(
                samples.pos,
                k_smpls,
                point_estimates=point_estimates,
                **draw_linear_kwargs,
                **kwargs,
            )
            if sample_mode.lower().startswith("nonlinear"):
                samples, st_smpls = self.nonlinearly_update_samples(
                    samples,
                    point_estimates=point_estimates,
                    **nonlinearly_update_kwargs,
                    **kwargs,
                )
            elif not sample_mode.lower().startswith("linear"):
                ve = f"invalid sampling mode {sample_mode!r}"
                raise ValueError(ve)
        elif sample_mode.lower() == "nonlinear_update":
            samples, st_smpls = self.nonlinearly_update_samples(
                samples,
                point_estimates=point_estimates,
                **nonlinearly_update_kwargs,
                **kwargs,
            )
        elif sample_mode == "":
            samples, st_smpls = samples, 0  # Do nothing for MAP
        else:
            ve = f"invalid sampling mode {sample_mode!r}"
            raise ValueError(ve)
        return samples, st_smpls

    def kl_minimize(
        self,
        samples: Samples,
        minimize: Callable[..., optimize.OptimizeResults] = optimize._static_newton_cg,
        minimize_kwargs={},
        constants=(),
        **kwargs,
    ) -> optimize.OptimizeResults:
        fun_and_grad = Partial(
            self.kl_value_and_grad, primals_samples=samples, **kwargs
        )
        hessp = Partial(self.kl_metric, primals_samples=samples, **kwargs)
        pl = samples.pos
        if constants:
            from .likelihood import _parse_point_estimates, partial_insert_and_remove
            from .tree_math import Vector, zeros_like

            insert_axes, pl, primals_frozen = _parse_point_estimates(constants, pl)
            unflatten = Vector if insert_axes else None
            fun_and_grad = partial_insert_and_remove(
                fun_and_grad,
                insert_axes=(insert_axes,),
                flat_fill=(primals_frozen,),
                remove_axes=(False, insert_axes),
                unflatten=lambda x: (x[0], unflatten(x[1:])),
            )
            hessp = partial_insert_and_remove(
                hessp,
                insert_axes=(insert_axes, insert_axes),
                flat_fill=(primals_frozen, zeros_like(primals_frozen)),
                remove_axes=insert_axes,
                unflatten=unflatten,
            )
        kl_opt_state = minimize(
            None,
            x0=pl,
            fun_and_grad=fun_and_grad,
            hessp=hessp,
            **minimize_kwargs,
        )
        if constants:
            insert = partial_insert_and_remove(
                lambda x: x,
                insert_axes=(insert_axes,),
                flat_fill=(primals_frozen,),
                remove_axes=None,
                unflatten=None,
            )
            kl_opt_state = kl_opt_state._replace(
                x=insert(kl_opt_state.x), jac=insert(kl_opt_state.jac)
            )
        return kl_opt_state

    def init_state(
        self,
        key,
        *,
        nit=0,
        n_samples: Union[int, Callable[[int], int]],
        draw_linear_kwargs: DICT_OR_CALL4DICT_TYP = dict(
            cg_name="SL", cg_kwargs=dict()
        ),
        nonlinearly_update_kwargs: DICT_OR_CALL4DICT_TYP = dict(
            minimize_kwargs=dict(name="SN", cg_kwargs=dict(name="SNCG"))
        ),
        kl_kwargs: DICT_OR_CALL4DICT_TYP = dict(
            minimize_kwargs=dict(name="M", cg_kwargs=dict(name="MCG"))
        ),
        sample_mode: SMPL_MODE_GENERIC_TYP = "nonlinear_resample",
        point_estimates=(),
        constants=(),
    ) -> OptimizeVIState:
        """Initialize the state of the (otherwise state-less) VI approximation.

        Parameters
        ----------
        key : jax random number generation key
        nit : int
            Current iteration number.
        n_samples : int or callable
            Number of samples to draw.
        draw_linear_kwargs : dict or callable
            Configuration for drawing linear samples, see
            :func:`draw_linear_residual`.
        nonlinearly_update_kwargs : dict or callable
            Configuration for nonlinearly updating samples, see
            :func:`nonlinearly_update_residual`.
        kl_kwargs : dict or callable
            Keyword arguments for the KL minimizer.
        sample_mode : str or callable
            One in {"linear_sample", "linear_resample", "nonlinear_sample",
            "nonlinear_resample", "nonlinear_update"}. The mode denotes the way
            samples are drawn and/or updates, "linear" draws MGVI samples,
            "nonlinear" draws MGVI samples which are then nonlinearly updated
            with geoVI, the "_sample" versus "_resample" suffix denotes whether
            the same stochasticity or new stochasticity is used for the drawing
            of the samples, and "nonlinear_update" nonlinearly updates existing
            samples using geoVI.
        point_estimates: tree-like structure or tuple of str
            Pytree of same structure as likelihood input but with boolean
            leaves indicating whether to sample the value in the input or use
            it as a point estimate. As a convenience method, for dict-like
            inputs, a tuple of strings is also valid. From these the boolean
            indicator pytree is automatically constructed.
        constants: tree-like structure or tuple of str
            Pytree of same structure as likelihood input but with boolean
            leaves indicating whether to keep these values constant during the
            KL minimization. As a convenience method, for dict-like inputs, a
            tuple of strings is also valid. From these the boolean indicator
            pytree is automatically constructed.

        Most of the parameters can be callable, in which case they are called
        with the current iteration number as argument and should return the
        value to use for the current iteration.
        """
        config = dict(
            n_samples=n_samples,
            sample_mode=sample_mode,
            point_estimates=point_estimates,
            constants=constants,
            draw_linear_kwargs=draw_linear_kwargs,
            nonlinearly_update_kwargs=nonlinearly_update_kwargs,
            kl_kwargs=kl_kwargs,
        )
        return OptimizeVIState(nit, key, config=config)

    def update(
        self,
        samples: Samples,
        state: OptimizeVIState,
        /,
        **kwargs,
    ) -> tuple[Samples, OptimizeVIState]:
        """Moves the VI approximation one sample update and minimization forward.

        Parameters
        ----------
        samples : :class:`Samples`
            Current samples.
        state : :class:`OptimizeVIState`
            Current state of the VI approximation.
        kwargs : dict
            Keyword arguments passed to the residual sampling functions.
        """
        assert isinstance(samples, Samples)
        assert isinstance(state, OptimizeVIState)
        nit, key, config = state.nit, state.key, state.config

        sample_mode = _getitem_at_nit(config, "sample_mode", nit)
        point_estimates = _getitem_at_nit(config, "point_estimates", nit)
        constants = _getitem_at_nit(config, "constants", nit)
        n_samples = _getitem_at_nit(config, "n_samples", nit)
        draw_linear_kwargs = _getitem_at_nit(config, "draw_linear_kwargs", nit)
        nonlinearly_update_kwargs = _getitem_at_nit(
            config, "nonlinearly_update_kwargs", nit
        )
        # Make the `key` tick independently of whether samples are drawn or not
        key, sk = random.split(key, 2)
        samples, st_smpls = self.draw_samples(
            samples,
            key=sk,
            sample_mode=sample_mode,
            point_estimates=point_estimates,
            n_samples=n_samples,
            draw_linear_kwargs=draw_linear_kwargs,
            nonlinearly_update_kwargs=nonlinearly_update_kwargs,
            **kwargs,
        )

        kl_kwargs = _getitem_at_nit(config, "kl_kwargs", nit).copy()
        kl_opt_state = self.kl_minimize(
            samples, constants=constants, **kl_kwargs, **kwargs
        )
        samples = samples.at(kl_opt_state.x)
        # Remove unnecessary references
        kl_opt_state = kl_opt_state._replace(x=None, jac=None, hess=None, hess_inv=None)

        state = state._replace(
            nit=nit + 1,
            key=key,
            sample_state=st_smpls,
            minimization_state=kl_opt_state,
        )
        return samples, state

    def run(self, samples, *args, **kwargs) -> tuple[Samples, OptimizeVIState]:
        state = self.init_state(*args, **kwargs)
        nm = self.__class__.__name__
        for i in range(state.nit, self.n_total_iterations):
            logger.info(f"{nm}: Starting {i + 1:04d}")
            samples, state = self.update(samples, state)
            msg = self.get_status_message(
                samples, state, map=self.residual_map, name=nm
            )
            logger.info(msg)
        return samples, state


def optimize_kl(
    likelihood: Likelihood,
    position_or_samples,
    *,
    key,
    n_total_iterations: int,
    n_samples,
    point_estimates=(),
    constants=(),
    kl_jit=True,
    residual_jit=True,
    kl_map=jax.vmap,
    residual_map="lmap",
    kl_reduce=_reduce,
    mirror_samples=True,
    draw_linear_kwargs=dict(cg_name="SL", cg_kwargs=dict()),
    nonlinearly_update_kwargs=dict(
        minimize_kwargs=dict(name="SN", cg_kwargs=dict(name="SNCG"))
    ),
    kl_kwargs=dict(minimize_kwargs=dict(name="M", cg_kwargs=dict(name="MCG"))),
    sample_mode: SMPL_MODE_GENERIC_TYP = "nonlinear_resample",
    resume: Union[str, bool] = False,
    callback: Optional[Callable[[Samples, OptimizeVIState], None]] = None,
    odir: Optional[str] = None,
    devices: Optional[list] = None,
    kl_device_map="shard_map",
    residual_device_map="pmap",
    _optimize_vi=None,
    _optimize_vi_state=None,
) -> tuple[Samples, OptimizeVIState]:
    """One-stop-shop for MGVI/geoVI style VI approximation.

    Parameters
    ----------
    position_or_samples: Samples or tree-like
        Initial position for minimization.
    resume : str or bool
        Resume partially run optimization. If `True`, the optimization is
        resumed from the previos state in `odir` otherwise it is resumed from
        the location toward which `resume` points.
    callback : callable or None
        Function called after every global iteration taking the samples and the
        optimization state.
    odir : str or None
        Path at which all output files are saved.


    See :class:`OptimizeVI` and :func:`OptimizeVI.init_state` for the remaining
    parameters and further details on the optimization.
    """
    LAST_FILENAME = "last.pkl"
    MINISANITY_FILENAME = "minisanity.txt"

    opt_vi = _optimize_vi if _optimize_vi is not None else None
    if opt_vi is None:
        opt_vi = OptimizeVI(
            likelihood,
            n_total_iterations=n_total_iterations,
            kl_jit=kl_jit,
            residual_jit=residual_jit,
            kl_map=kl_map,
            residual_map=residual_map,
            kl_reduce=kl_reduce,
            mirror_samples=mirror_samples,
            devices=devices,
            kl_device_map=kl_device_map,
            residual_device_map=residual_device_map,
        )

    last_fn = os.path.join(odir, LAST_FILENAME) if odir is not None else None
    resume_fn = resume if os.path.isfile(resume) else last_fn
    sanity_fn = os.path.join(odir, MINISANITY_FILENAME) if odir is not None else None

    samples = None
    if isinstance(position_or_samples, Samples):
        samples = position_or_samples
    else:
        samples = Samples(pos=position_or_samples, samples=None, keys=None)
    opt_vi_st = None
    if resume and os.path.isfile(resume_fn):
        if samples.pos is not None:
            logger.warning("overwriting `position_or_samples` with `resume`")
        with open(resume_fn, "rb") as f:
            samples, opt_vi_st = pickle.load(f)

    opt_vi_st_init = opt_vi.init_state(
        key,
        n_samples=n_samples,
        draw_linear_kwargs=draw_linear_kwargs,
        nonlinearly_update_kwargs=nonlinearly_update_kwargs,
        kl_kwargs=kl_kwargs,
        sample_mode=sample_mode,
        point_estimates=point_estimates,
        constants=constants,
    )
    opt_vi_st = _optimize_vi_state if _optimize_vi_state is not None else opt_vi_st
    opt_vi_st = opt_vi_st_init if opt_vi_st is None else opt_vi_st
    if len(opt_vi_st.config) == 0:  # resume or _optimize_vi_state has empty config
        opt_vi_st = opt_vi_st._replace(config=opt_vi_st_init.config)

    if odir:
        makedirs(odir, exist_ok=True)
    if not resume and sanity_fn is not None:
        with open(sanity_fn, "w"):
            pass

    nm = "OPTIMIZE_KL"
    for i in range(opt_vi_st.nit, opt_vi.n_total_iterations):
        logger.info(f"{nm}: Starting {i + 1:04d}")
        samples, opt_vi_st = opt_vi.update(samples, opt_vi_st)
        msg = opt_vi.get_status_message(samples, opt_vi_st, name=nm)
        logger.info(msg)
        if sanity_fn is not None:
            with open(sanity_fn, "a") as f:
                f.write("\n" + msg)
        if last_fn is not None:
            with open(last_fn, "wb") as f:
                # TODO: Make all arrays numpy arrays as to not instantiate on
                # the main device when loading
                pickle.dump((samples, opt_vi_st._replace(config={})), f)
        if callback is not None:
            callback(samples, opt_vi_st)

    return samples, opt_vi_st
