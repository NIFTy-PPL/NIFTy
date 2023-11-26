#!/usr/bin/env python3

# SPDX-License-Identifier: BSD-2-Clause
# Authors: Philipp Frank, Jakob Roth, Gordian Edenhofer

import inspect
import os
import pickle
from functools import partial
from os import makedirs
from pickle import PicklingError
from typing import Any, Callable, Literal, NamedTuple, Optional, TypeVar, Union

import jax
import numpy as np
from jax import numpy as jnp
from jax import random, tree_map
from jax.tree_util import Partial

from . import optimize
from .evi import Samples, _parse_jit, curve_residual, draw_linear_residual
from .likelihood import Likelihood, StandardHamiltonian
from .logger import logger
from .minisanity import minisanity
from .tree_math import get_map, hide_strings

P = TypeVar("P")


def get_status_message(
    samples, state, residual=None, *, name="", map="lmap"
) -> str:
    energy = state.minimization_state.fun
    msg_smpl = ""
    if isinstance(state.sample_state, optimize.OptimizeResults):
        nlsi = tuple(int(el) for el in state.sample_state.nit)
        msg_smpl = f"\n#(Nonlinear Sampling Steps) {nlsi}"
    elif isinstance(state.sample_state, (np.ndarray, jax.Array)):
        nlsi = tuple(int(el) for el in state.sample_state)
        msg_smpl = f"\nLinear Sampling Status {nlsi}"
    mini_res = ""
    if residual is not None:
        _, mini_res = minisanity(samples, residual, map=map)
    _, mini_pr = minisanity(samples, map=map)
    msg = (
        f"Post {name}: Iteration {state.nit - 1:04d} ⛰:{energy:+2.4e}"
        f"{msg_smpl}"
        f"\n#(KL minimization steps) {state.minimization_state.nit}"
        f"\nLikelihood residual(s):\n{mini_res}"
        f"\nPrior residual(s):\n{mini_pr}"
        f"\n"
    )
    return msg


_reduce = partial(tree_map, partial(jnp.mean, axis=0))


def _kl_vg(
    likelihood,
    primals,
    primals_samples,
    *,
    map=jax.vmap,
    reduce=_reduce,
):
    map = get_map(map)

    assert isinstance(primals_samples, Samples)
    ham = StandardHamiltonian(likelihood=likelihood)
    vvg = map(jax.value_and_grad(ham))
    s = vvg(primals_samples.at(primals).samples)
    return reduce(s)


def _kl_met(
    likelihood,
    primals,
    tangents,
    primals_samples,
    *,
    map=jax.vmap,
    reduce=_reduce
):
    map = get_map(map)

    assert isinstance(primals_samples, Samples)
    ham = StandardHamiltonian(likelihood=likelihood)
    vmet = map(ham.metric, in_axes=(0, None))
    s = vmet(primals_samples.at(primals).samples, tangents)
    return reduce(s)


@jax.jit
def concatenate_zip(*arrays):
    return tree_map(
        lambda *x: jnp.stack(x, axis=1).reshape((-1, ) + x[0].shape[1:]),
        *arrays
    )


SMPL_DO_TYP = Literal[None, "resample_mgvi", "resample_geovi", "curve"]
SMPL_DO_GEN_TYP = Union[SMPL_DO_TYP, Callable[[int], SMPL_DO_TYP]]
DICT_OR_CALL4DICT_TYP = Union[Callable[[int], dict], dict]


class OptimizeVIState(NamedTuple):
    nit: int
    key: Any
    sample_instruction: SMPL_DO_GEN_TYP
    sample_state: Optional[optimize.OptimizeResults]
    minimization_state: Optional[optimize.OptimizeResults]
    config: dict[str, Union[dict, Callable[[int], Any]]]


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
    distribution in model space, or non-linear, i.e. following a standard normal
    distribution in the canonical coordinate system of the Riemannian manifold
    associated with the metric of the approximate posterior distribution. The
    coordinate transformation for the non-linear sample is approximated by an
    expansion.

    Both linear and non-linear sample start by drawing a sample from the
    inverse metric. To do so, we draw a sample which has the metric as
    covariance structure and apply the inverse metric to it. The sample
    transformed in this way has the inverse metric as covariance. The first
    part is trivial since we can use the left square root of the metric
    :math:`L` associated with every likelihood:
    .. math::
        \tilde{d} \leftarrow \mathcal{G}(0,\mathbb{1}) \\
        t = L \tilde{d}
    with :math:`t` now having a covariance structure of
    .. math::
        <t t^\dagger> = L <\tilde{d} \tilde{d}^\dagger> L^\dagger = M .

    To transform the sample to an inverse sample, we apply the inverse
    metric. We can do so using the conjugate gradient algorithm (CG). The CG
    algorithm yields the solution to :math:`M s = t`, i.e. applies the
    inverse of :math:`M` to :math:`t`:
    .. math::
        M &s =  t \\
        &s = M^{-1} t = cg(M, t) .
    The linear sample is :math:`s`.

    The non-linear sampling uses :math:`s` as a starting value and curves it in
    a non-linear way as to better resemble the posterior locally. See the below
    reference literature for more details on the non-linear sampling.

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
        point_estimates=(),
        constants=(),  # TODO
        kl_jit=True,
        residual_jit=True,
        kl_map=jax.vmap,
        residual_map="lmap",
        kl_reduce=_reduce,
        mirror_samples=True,
        _kl_value_and_grad: Optional[Callable] = None,
        _kl_metric: Optional[Callable] = None,
        _draw_linear_residual: Optional[Callable] = None,
        _curve_residual: Optional[Callable] = None,
        _get_status_message: Optional[Callable] = None,
    ):
        """JaxOpt style minimizer for a VI approximation of a distribution with
        samples.

        Parameters:
        ----------
        likelihood: :class:`~nifty8.re.likelihood.Likelihood`
            Likelihood to be used for inference.
        n_total_iterations: int
            Total number of iterations. One iteration consists of the steps
            1) - 3).
        point_estimates: tree-like structure or tuple of str
            Pytree of same structure as likelihood input but with boolean
            leaves indicating whether to sample the value in the input or use
            it as a point estimate. As a convenience method, for dict-like
            inputs, a tuple of strings is also valid. From these the boolean
            indicator pytree is automatically constructed.
        constants: tree-like structure or tuple of str
            Not implemented yet, sorry :( Do bug me (Gordian) at
            edh@mpa-garching.mpg.de if you wanted to run with this option.
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

        Notes:
        ------
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

        if not (constants == () or constants is None):
            raise NotImplementedError()
        if mirror_samples is False:
            raise NotImplementedError()

        if _kl_value_and_grad is None:
            _kl_value_and_grad = kl_jit(
                partial(_kl_vg, likelihood, map=kl_map, reduce=kl_reduce)
            )
        if _kl_metric is None:
            _kl_metric = kl_jit(
                partial(_kl_met, likelihood, map=kl_map, reduce=kl_reduce)
            )
        if _draw_linear_residual is None:
            _draw_linear_residual = residual_jit(
                partial(
                    draw_linear_residual,
                    likelihood,
                    point_estimates=point_estimates,
                )
            )
        if _curve_residual is None:
            # TODO: Pull out `jit` from `curve_residual` once NCG is jit-able
            # TODO: STOP inserting `point_estimes` and instead defer it to `update`
            from .evi import _curve_residual_functions

            _curve_funcs = _curve_residual_functions(
                likelihood=likelihood,
                point_estimates=point_estimates,
                jit=residual_jit,
            )
            _curve_residual = partial(
                curve_residual,
                likelihood,
                point_estimates=point_estimates,
                _curve_funcs=_curve_funcs,
            )
        if _get_status_message is None:
            _get_status_message = partial(
                get_status_message,
                residual=likelihood.normalized_residual,
                name=self.__class__.__name__,
            )

        self.n_total_iterations = None
        self.kl_value_and_grad = None
        self.kl_metric = None
        self.draw_linear_residual = None
        self.curve_residual = None
        self.residual_map = None
        self.get_status_message = None
        self.replace(
            n_total_iterations=n_total_iterations,
            kl_value_and_grad=_kl_value_and_grad,
            kl_metric=_kl_metric,
            draw_linear_residual=_draw_linear_residual,
            curve_residual=_curve_residual,
            get_status_message=_get_status_message,
            residual_map=residual_map,
        )

    def replace(
        self,
        *,
        n_total_iterations=None,
        kl_value_and_grad=None,
        kl_metric=None,
        draw_linear_residual=None,
        curve_residual=None,
        get_status_message=None,
        residual_map=None,
    ):
        self.n_total_iterations = n_total_iterations if n_total_iterations is None else n_total_iterations
        self.kl_value_and_grad = kl_value_and_grad if kl_value_and_grad is not None else self.kl_value_and_grad
        self.kl_metric = kl_metric if kl_metric is not None else self.kl_metric
        self.draw_linear_residual = draw_linear_residual if draw_linear_residual is not None else self.draw_linear_residual
        self.curve_residual = curve_residual if curve_residual is not None else self.curve_residual
        self.residual_map = residual_map if residual_map is not None else self.residual_map
        self.get_status_message = get_status_message if get_status_message is not None else self.get_status_message

    def draw_linear_samples(self, primals, keys, **kwargs):
        # NOTE, use `Partial` in favor of `partial` to allow the (potentially)
        # re-jitting `residual_map` to trace the kwargs
        kwargs = hide_strings(kwargs)
        sampler = Partial(self.draw_linear_residual, **kwargs)
        sampler = self.residual_map(sampler, in_axes=(None, 0))
        smpls, smpls_states = sampler(primals, keys)
        # zip samples such that the mirrored-counterpart always comes right
        # after the original sample
        smpls = Samples(
            pos=primals, samples=concatenate_zip(smpls, -smpls), keys=keys
        )
        return smpls, smpls_states

    def curve_samples(self, samples: Samples, **kwargs):
        # NOTE, use `Partial` in favor of `partial` to allow the (potentially)
        # re-jitting `residual_map` to trace the kwargs
        kwargs = hide_strings(kwargs)
        curver = Partial(self.curve_residual, **kwargs)
        curver = self.residual_map(curver, in_axes=(None, 0, 0, 0))
        assert len(samples.keys) == len(samples) // 2
        metric_sample_key = concatenate_zip(*((samples.keys, ) * 2))
        sgn = jnp.ones(len(samples.keys))
        sgn = concatenate_zip(sgn, -sgn)
        smpls, smpls_states = curver(
            samples.pos, samples._samples, metric_sample_key, sgn
        )
        smpls = Samples(pos=samples.pos, samples=smpls, keys=samples.keys)
        return smpls, smpls_states

    def init_state(
        self,
        key,
        *,
        nit=0,
        n_samples: Union[int, Callable[[int], int]],
        draw_linear_samples: DICT_OR_CALL4DICT_TYP = dict(
            cg_name="SL", cg_kwargs=dict()
        ),
        curve_samples: DICT_OR_CALL4DICT_TYP = dict(
            minimize_kwargs=dict(name="SN", cg_kwargs=dict(name="SNCG"))
        ),
        minimize: Callable[
            ...,
            optimize.OptimizeResults,
        ] = optimize._newton_cg,
        minimize_kwargs: DICT_OR_CALL4DICT_TYP = dict(
            name="M", cg_kwargs=dict(name="MCG")
        ),
        sample_instruction: SMPL_DO_GEN_TYP = "resample_geovi",
    ) -> OptimizeVIState:
        """Initialize the state of the (otherwise state-less) VI approximation.

        Parameters
        ----------
        key : jax random number generataion key
        nit : int
            Current iteration number.
        n_samples : int or callable
            Number of samples to draw.
        draw_linear_samples : dict or callable
            Configuration for drawing linear samples, see
            :func:`draw_linear_residual`.
        curve_samples : dict or callable
            Configuration for curving samples, see :func:`curve_residual`.
        minimize : callable
            Minimizer to use for the KL minimization.
        minimize_kwargs : dict or callable
            Keyword arguments for the minimizer.
        sample_instruction : str or callable
            One in {"resample_mgvi", "resample_geovi", "curve"}.

        Most of the parameters can be callable, in which case they are called
        with the current iteration number as argument and should return the
        value to use for the current iteration.
        """
        config = {
            "n_samples": n_samples,
            "draw_linear_samples": draw_linear_samples,
            "curve_samples": curve_samples,
            "minimize": minimize,
            "minimize_kwargs": minimize_kwargs,
        }
        sample_instruction = sample_instruction
        state = OptimizeVIState(
            nit,
            key,
            sample_instruction=sample_instruction,
            sample_state=None,
            minimization_state=None,
            config=config
        )
        return state

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
        nit = state.nit + 1
        key = state.key
        st_smpls = state.sample_state
        config = state.config

        smpls_do = state.sample_instruction
        smpls_do: str = smpls_do(nit) if callable(smpls_do) else smpls_do
        n_samples = _getitem_at_nit(config, "n_samples", nit)
        # Always resample if `n_samples` increased
        if n_samples > len(samples) and smpls_do.lower() == "curve":
            smpls_do = "resample_geovi"
        if smpls_do.lower().startswith("resample"):
            key, *k_smpls = random.split(key, n_samples + 1)
            k_smpls = jnp.array(k_smpls)
            kw = _getitem_at_nit(config, "draw_linear_samples", nit)
            samples, st_smpls = self.draw_linear_samples(
                samples.pos, k_smpls, **kw, **kwargs
            )
            if smpls_do.lower() == "resample_geovi":
                kw = _getitem_at_nit(config, "curve_samples", nit)
                samples, st_smpls = self.curve_samples(samples, **kw, **kwargs)
            elif smpls_do.lower() != "resample_mgvi":
                ve = f"invalid resampling instruction {smpls_do}"
                raise ValueError(ve)
        elif smpls_do.lower() == "curve":
            kw = _getitem_at_nit(config, "curve_samples", nit)
            samples, st_smpls = self.curve_samples(samples, **kw, **kwargs)
        elif smpls_do is None:
            ve = f"invalid resampling instruction {smpls_do}"
            raise ValueError(ve)

        minimize = _getitem_at_nit(config, "minimize", nit)
        kw = {
            "fun_and_grad":
                partial(self.kl_value_and_grad, primals_samples=samples),
            "hessp":
                partial(self.kl_metric, primals_samples=samples),
        }
        kw |= _getitem_at_nit(config, "minimize_kwargs", nit)
        kl_opt_state = minimize(None, x0=samples.pos, **kw)
        samples = samples.at(kl_opt_state.x)
        # Remove unnecessary references
        kl_opt_state = kl_opt_state._replace(
            x=None, jac=None, hess=None, hess_inv=None
        )

        state = state._replace(
            nit=nit,
            key=key,
            sample_state=st_smpls,
            minimization_state=kl_opt_state,
        )
        return samples, state

    def run(self, samples, *args, **kwargs) -> tuple[Samples, OptimizeVIState]:
        state = self.init_state(*args, **kwargs)
        for i in range(self.n_total_iterations):
            logger.info(f"{self.__class__.__name__} :: {i:04d}")
            samples, state = self.update(samples, state)
            msg = self.get_status_message(samples, state, map=self.residual_map)
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
    draw_linear_samples=dict(cg_name="SL", cg_kwargs=dict()),
    curve_samples=dict(
        minimize_kwargs=dict(name="SN", cg_kwargs=dict(name="SNCG"))
    ),
    minimize: Callable[
        ...,
        optimize.OptimizeResults,
    ] = optimize._newton_cg,
    minimize_kwargs=dict(name="M", cg_kwargs=dict(name="MCG")),
    sample_instruction: SMPL_DO_GEN_TYP = "resample_geovi",
    resume: Union[str, bool] = False,
    callback: Optional[Callable[[Samples, OptimizeVIState], None]] = None,
    odir: Optional[str] = None,
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
            point_estimates=point_estimates,
            constants=constants,
            kl_jit=kl_jit,
            residual_jit=residual_jit,
            kl_map=kl_map,
            residual_map=residual_map,
            kl_reduce=kl_reduce,
            mirror_samples=mirror_samples
        )

    samples = None
    if isinstance(position_or_samples, Samples):
        samples = position_or_samples
    else:
        samples = Samples(pos=position_or_samples, samples=None, keys=None)

    last_fn = os.path.join(odir, LAST_FILENAME) if odir is not None else None
    resume_fn = resume if os.path.isfile(resume) else last_fn
    if _optimize_vi_state is not None:
        opt_vi_state = _optimize_vi_state
    elif resume and os.path.isfile(resume_fn):
        if not os.path.isfile(resume) or not odir:
            ve = "unable to resume without `resume` or `odir` being a path"
            raise ValueError(ve)
        with open(resume_fn, "rb") as f:
            samples, opt_vi_state = pickle.load(f)
    else:
        opt_vi_state = opt_vi.init_state(
            key,
            n_samples=n_samples,
            draw_linear_samples=draw_linear_samples,
            curve_samples=curve_samples,
            minimize=minimize,
            minimize_kwargs=minimize_kwargs,
            sample_instruction=sample_instruction,
        )
    sanity_fn = os.path.join(
        odir, MINISANITY_FILENAME
    ) if odir is not None else None

    # Test if the state is pickle-able to avoid an unfortunate end
    try:
        pickle.dumps(opt_vi_state)
    except PicklingError as e:
        ve = "state not pickle-able; check configuration for e.g. `lambda`s"
        raise ValueError(ve) from e

    if odir:
        makedirs(odir, exist_ok=True)

    for i in range(opt_vi_state.nit, opt_vi.n_total_iterations):
        logger.info(f"OPTIMIZE_KL Iteration {i:04d}")
        samples, opt_vi_state = opt_vi.update(samples, opt_vi_state)
        msg = opt_vi.get_status_message(samples, opt_vi_state)
        logger.info(msg)
        if sanity_fn is not None:
            with open(sanity_fn, "a") as f:
                f.write("\n" + msg)
        if last_fn is not None:
            with open(last_fn, "wb") as f:
                pickle.dump((samples, opt_vi_state), f)
        if callback != None:
            callback(samples, opt_vi_state)

    return samples, opt_vi_state
