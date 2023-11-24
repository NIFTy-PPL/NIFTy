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
    msg_nonlin = ""
    if state.sample_state is not None:
        nlsi = tuple(int(el) for el in state.sample_state.nit)
        msg_nonlin = f"\n#(Nonlinear Sampling Steps) {nlsi}"
    mini_res = ""
    if residual is not None:
        _, mini_res = minisanity(samples, residual, map=map)
    _, mini_pr = minisanity(samples, map=map)
    msg = (
        f"Post {name}: Iteration {state.nit - 1:04d} ⛰:{energy:+2.4e}"
        f"{msg_nonlin}"
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


_smpl_do_typ = Literal[None, "resample_mgvi", "resample_geovi", "curve"]


class OptimizeVIState(NamedTuple):
    nit: int
    key: Any
    sample_instruction: Union[_smpl_do_typ, Callable[[int], _smpl_do_typ]]
    sample_state: Optional[optimize.OptimizeResults]
    minimization_state: Optional[optimize.OptimizeResults]
    config: dict[str, Union[dict, Callable[[int], Any]]]


def _getitem_at_nit(config, key, nit):
    c = config[key]
    if callable(c) and len(inspect.getfullargspec(c).args) == 1:
        return c(nit)
    return c


class OptimizeVI:
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
        -----------
        n_iter: int
            Total number of iterations. One iteration consists of the steps
            1) - 3).
        kl_solver: Callable
            Solver that minimizes the KL w.r.t. the mean of the samples.
        sample_generator: Callable
            Function to generate new samples.
        sample_update: Callable
            Function to update existing samples.
        kl_solver_kwargs: dict
            Optional keyword arguments to be passed on to `kl_solver`. They are
            added to the optimizers state and passed on at each `update` step.
        sample_generator_kwargs: dict
            Optional keyword arguments to be passed on to `sample_generator`.
        sample_update_kwargs: dict
            Optional keyword arguments to be passed on to `sample_update`.

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

        The functions `kl_solver`, `sample_generator`, and `sample_update` all
        share the same syntax: They must take two inputs, samples and keys,
        where keys are the jax.random keys that are used for the samples.
        Additionally they each can take respective keyword arguments. These
        are passed on at runtime and stored in the optimizers state. All
        functions must return samples, as an instance of `Samples`

        TODO:
        MGVI/geoVI interface that creates the input functions of `OptimizeVI`
        from a `Likelihood`.
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
        The linear sample is :math:`s`. The non-linear sample uses :math:`s` as
        a starting value and curves it in a non-linear way as to better resemble
        the posterior locally. See the below reference literature for more
        details on the non-linear sampling.

        Parameters
        ----------
        likelihood : :class:`nifty8.re.likelihood.Likelihood`
            Likelihood to be used for inference.
        n_iter : int
            Number of iterations.
        point_estimates : tree-like structure or tuple of str
            Pytree of same structure as likelihood input but with boolean leaves
            indicating whether to sample the value in the input or use it as a
            point estimate. As a convenience method, for dict-like inputs, a
            tuple of strings is also valid. From these the boolean indicator
            pytree is automatically constructed.
        kl_kwargs: dict
            Keyword arguments passed on to `kl_solver`. Can be used to specify the
            jit and map behavior of the function being constructed.
        linear_sampling_kwargs: dict
            Keyword arguments passed on to `linear_residual_sampler`. Includes
            the cg config used for linear sampling and its jit/map configuration.
        curve_kwargs: dict
            Keyword arguments passed on to `curve_sampler`. Can be used to specify
            the jit and map behavior of the function being constructed.
        _raise_notconverged: bool
            Whether to raise inversion & minimization errors during sampling.
            Default is False.
        See also
        --------
        `Geometric Variational Inference`, Philipp Frank, Reimar Leike,
        Torsten A. Enßlin, `<https://arxiv.org/abs/2105.10470>`_
        `<https://doi.org/10.3390/e23070853>`_
        `Metric Gaussian Variational Inference`, Jakob Knollmüller,
        Torsten A. Enßlin, `<https://arxiv.org/abs/1901.11033>`_
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
        n_samples,
        draw_linear_samples=dict(cg_name="SL", cg_kwargs=dict()),
        curve_samples=dict(
            minimize_kwargs=dict(name="SN", cg_kwargs=dict(name="SNCG"))
        ),
        minimize: Callable[
            ...,
            optimize.OptimizeResults,
        ] = optimize._newton_cg,
        minimize_kwargs=dict(name="M", cg_kwargs=dict(name="MCG")),
        sample_instruction="resample_geovi",
    ) -> OptimizeVIState:
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
        """One sampling and kl optimization step."""
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
        smpls_do = "resample_geovi" if n_samples > len(samples) else smpls_do
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
            elif smpls_do.lower() == "resample_mgvi":
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
    sample_instruction="resample_geovi",
    resume: Union[str, bool] = False,
    callback: Optional[Callable[[Samples, OptimizeVIState], None]] = None,
    odir=None,
    _optimize_vi=None,
    _optimize_vi_state=None,
) -> tuple[Samples, OptimizeVIState]:
    """Interface for KL minimization similar to NIFTy optimize_kl.

    Parameters
    ----------
    likelihood : :class:`nifty8.re.likelihood.Likelihood` or callable
        Likelihood to be used for inference. If its a callable, must be of the
        form f(current_iteration) -> `Likelihood`. Allows to use different
        likelihoods during minimization.
    pos : Initial position for minimization.
    total_iterations : int
        Number of resampling loops.
    n_samples : int or callable
        Number of samples used to sample Kullback-Leibler divergence. See
        `likelihood` for the callable convention.
    key : jax random number generataion key
    point_estimates : tree-like structure or tuple of str
        Pytree of same structure as `pos` but with boolean leaves indicating
        whether to sample the value in `pos` or use it as a point estimate. As
        a convenience method, for dict-like `pos`, a tuple of strings is also
        valid. From these the boolean indicator pytree is automatically
        constructed.
    sampling_method: str or callable
        Sampling method used for vi approximation. Default is `altmetric`.
    make_kl_kwargs: dict or callable
        Configuration of the KL optimizer passed on to `optimizeVI_callables`.
        Can also be a function of iteration number, in which case
        `optimizeVI_callables` is called again to create new solvers. Note that
        this may trigger re-compilations! The config of the minimizer used in
        the kl optimization can be set at runtime via `kl_solver_kwargs`.
    make_sample_generator_kwargs: dict or callable
        Configuration of the sample generator `linear_sampling` passed on to
        `optimizeVI_callables`. Can also be a function of iteration number.
    make_sample_update_kwargs:  dict or callable
        Configuration of the sample update `curve` passed on to
        `optimizeVI_callables`. Can also be a function of iteration number.
    kl_solver_kwargs: dict or callable
        Keyword arguments to be passed on to `kl_solver` in `OptimizeVI`.
        Specifies the minimizer being used during the kl optimization step and
        its config. Can be a function of iteration number to change the
        minimizers configuration during runtime.
    sample_generator_kwargs: str or callable
        Keyword arguments to be passed on to `sample_generator` in `OptimizeVI`.
        Runtime configuration of the linear sampling.
    sample_update_kwargs: dict or callable
        Keyword arguments to be passed on to `sample_update` in `OptimizeVI`.
        Specifies the minimizer being used during the non-linear `curve` sample
        step and its config.
    resample: bool or callable
        Whether to resample with new random numbers or not. Default is False
    callback : callable or None
        Function that is called after every global iteration. It needs to be a
        function taking 3 arguments: 1. the current samples,
                                     2. the state of `OptimizeVI`,
                                     3. the global iteration number.
        Default: None.
    output_directory : str or None
        Directory in which all output files are saved. If None, no output is
        stored.  Default: None.
    resume : bool
        Resume partially run optimization. If `True` and `output_directory`
        is specified it resumes optimization. Default: False.
    verbosity : int
        Sets verbosity of optimization. If -1 only the current global
        optimization index is printed. If 0 CG steps of linear sampling,
        NewtonCG steps of non linear sampling and NewtonCG steps of KL
        optimization are printed. If set to 1 additionally the internal CG steps
        of the NewtonCG optimization are printed. Default: 0.
    _vi_callables: tuple of callable or callable (optional)
        Option to completely sidestep the `optimizeVI_callables` interface.
        Allows to specify a tuple of the three functions `kl_solver`,
        `sample_generator`, and `sample_update` that are used to instantiate
        `OptimizeVI`. If specified, these functions are used instead of the ones
        created by `optimizeVI_callables` and the corresonding arguments above
        are ignored. Can also be a function of iteration number instead.
    _update_state: callable (Default update_state)
        Function to update the state of `OptimizeVI` according to the config
        specified by the arguments above. The default `update_state` respects
        the MGVI/geoVI logic and implements the corresponding update. If
        `_vi_callables` is set, this may be changed to a different function that
        is applicable to the functions that are being passed on.
    """
    LAST_FILENAME = "last.pkl"

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
        if odir is not None:
            sanity_fn = os.path.join(odir, "minisanity")
            if os.path.isfile(sanity_fn) and opt_vi_state.nit != 0:
                with open(sanity_fn, "a") as f:
                    f.write("\n" + msg)
            if odir is not None:
                with open(last_fn, "wb") as f:
                    pickle.dump((samples, opt_vi_state), f)
        if callback != None:
            callback(samples, opt_vi_state)

    return samples, opt_vi_state
