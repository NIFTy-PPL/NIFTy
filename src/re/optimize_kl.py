#!/usr/bin/env python3
# Copyright(C) 2013-2023 Max-Planck-Society
# SPDX-License-Identifier: BSD-2-Clause
# Authors: Philipp Frank, Jakob Roth

import pickle
import sys
import jax
import jax.numpy as jnp
from os import makedirs
from os.path import isfile
from typing import Any, Callable, List, NamedTuple, Union
from functools import partial
from .optimize import OptimizeResults, minimize
from .tree_math.vector_math import dot, vdot
from .tree_math.forest_math import stack
from .tree_math.vector import Vector
from .likelihood import Likelihood, StandardHamiltonian
from .kl import Samples, _sample_linearly
from .smap import smap

def _ham_vg(likelihood, primals, primals_samples):
    assert isinstance(primals_samples, Samples)
    ham = StandardHamiltonian(likelihood=likelihood)
    vvg = jax.vmap(jax.value_and_grad(ham))
    s = vvg(primals_samples.at(primals).samples)
    return jax.tree_util.tree_map(partial(jnp.mean, axis=0), s)

def _ham_metric(likelihood, primals, tangents, primals_samples):
    assert isinstance(primals_samples, Samples)
    ham = StandardHamiltonian(likelihood=likelihood)
    vmet = jax.vmap(ham.metric, in_axes=(0, None))
    s = vmet(primals_samples.at(primals).samples, tangents)
    return jax.tree_util.tree_map(partial(jnp.mean, axis=0), s)

def _lh_trafo(likelihood, primals):
    return likelihood.transformation(primals)

def _nl_g(likelihood, p, lh_trafo_at_p, x):
    t = _lh_trafo(likelihood, x) - lh_trafo_at_p
    return x - p + likelihood.left_sqrt_metric(p, t)

def _nl_residual(likelihood, x, p, lh_trafo_at_p, ms_at_p):
    r = ms_at_p - _nl_g(likelihood, p, lh_trafo_at_p, x)
    return 0.5*dot(r, r)

def _nl_metric(likelihood, primals, tangents, p, lh_trafo_at_p):
    f = partial(_nl_g, likelihood, p, lh_trafo_at_p)
    _, jj = jax.jvp(f, (primals,), (tangents,))
    return jax.vjp(f, primals)[1](jj)[0]

def _nl_sampnorm(likelihood, natgrad, p):
    o = partial(likelihood.left_sqrt_metric, p)
    fpp = jax.linear_transpose(o, likelihood.lsm_tangents_shape)(natgrad)
    return jnp.sqrt(vdot(natgrad, natgrad) + vdot(fpp, fpp))

class OptVIState(NamedTuple):
  """Named tuple containing state information."""
  niter: int
  samples: Samples
  sampling_states: List[OptimizeResults]
  minimization_state: OptimizeResults


class OptimizeVI:
    def __init__(self,
                 likelihood: Union[Likelihood, None],
                 n_iter: int,
                 key: jax.random.PRNGKey,
                 n_samples: int,
                 sampling_method: str = 'altmetric',
                 sampling_minimizer = 'newtoncg',
                 sampling_kwargs: dict = {'xtol':0.01},
                 sampling_cg_kwargs: dict = {'maxiter':50},
                 minimizer: str = 'newtoncg',
                 minimization_kwargs: dict = {},
                 jit = jax.jit,
                 _lh_funcs: Any = None):
        # TODO reintroduce point-estimates (also possibly different sampling
        # methods for pytree)
        """JaxOpt style minimizer for VI approximation of a Bayesian inference
        problem assuming a standard normal prior distribution.

        Parameters
        ----------
        likelihood : :class:`nifty8.re.likelihood.Likelihood`
            Likelihood to be used for inference.
        n_iter : int
            Number of iterations.
        key : jax random number generataion key
        n_samples : int
            Number of samples used to sample Kullback-Leibler divergence. The
            samples get mirrored, so the actual number of samples used for the
            KL estimate is twice the number of `n_samples`.
        sampling_method: str
            Sampling method used for vi approximation. Default is `altmetric`.
        sampling_minimizer: str
            Minimization method used for non-linear sample minimization.
        sampling_kwargs: dict
            Keyword arguments for minimizer used for sample minimization.
        sampling_cg_kwargs: dict
            Keyword arguments for ConjugateGradient used for the linear part of
            sample minimization.
        minimizer: str or callable
            Minimization method used for KL minimization.
        minimization_kwargs : dict
            Keyword arguments for minimizer used for KL minimization.
        """
        self._n_iter = n_iter
        self._sampling_method = sampling_method
        if self._sampling_method not in ['linear', 'geometric', 'altmetric']:
            msg = f"Unknown sampling method: {self._sampling_method}"
            raise NotImplementedError(msg)
        self._minimizer = minimizer
        self._sampling_minimizer = sampling_minimizer
        self._sampling_kwargs = sampling_kwargs
        # Only use xtol for sampling since a custom gradient norm is used
        self._sampling_kwargs['absdelta'] = 0.
        self._mini_kwargs = minimization_kwargs
        self._keys = jax.random.split(key, n_samples)
        self._likelihood = likelihood
        self._n_samples = n_samples
        self._sampling_cg_kwargs = sampling_cg_kwargs

        if _lh_funcs is None:
            if likelihood is None:
                raise ValueError("Neither Likelihood nor funcs provided.")
            draw_metric = partial(_sample_linearly, likelihood,
                                  from_inverse=False)
            draw_metric = jax.vmap(draw_metric, in_axes=(None, 0),
                              out_axes=(None, 0))
            draw_linear = partial(_sample_linearly, likelihood,
                                  from_inverse = True,
                                  cg_kwargs = sampling_cg_kwargs)
            draw_linear = smap(draw_linear, in_axes=(None, 0))

            # KL funcs
            self._kl_vg = jit(partial(_ham_vg, likelihood))
            self._kl_metric = jit(partial(_ham_metric, likelihood))
            # Lin sampling
            self._draw_metric = jit(draw_metric)
            self._draw_linear = jit(draw_linear)
            # Non-lin sampling
            self._lh_trafo = jit(partial(_lh_trafo, likelihood))
            self._nl_vag = jit(jax.value_and_grad(
                               partial(_nl_residual, likelihood)))
            self._nl_metric = jit(partial(_nl_metric, likelihood))
            self._nl_sampnorm = jit(partial(_nl_sampnorm, likelihood))
        else:
            if likelihood is not None:
                msg = "Warning: Likelihood funcs is set, ignoring Likelihood"
                msg += " input"
                print(msg, file=sys.stderr)
            (self._kl_vg,
             self._kl_metric,
             self._draw_linear,
             self._draw_metric,
             self._lh_trafo,
             self._nl_vag,
             self._nl_metric,
             self._nl_sampnorm) = _lh_funcs

    @property
    def likelihood(self):
        return self._likelihood

    @property
    def n_samples(self):
        return self._n_samples

    @property
    def lh_funcs(self):
        return (self._kl_vg,
                self._kl_metric,
                self._draw_linear,
                self._draw_metric,
                self._lh_trafo,
                self._nl_vag,
                self._nl_metric,
                self._nl_sampnorm)

    def _linear_sampling(self, primals, from_inverse):
        if from_inverse:
            samples, met_smpls = self._draw_linear(primals, self._keys)
            samples = Samples(
                pos=primals,
                samples=jax.tree_map(lambda *x:
                                    jnp.concatenate(x), samples, -samples)
            )
        else:
            _, met_smpls = self._draw_metric(primals, self._keys)
            samples = None
        met_smpls = Samples(pos=None,
                            samples=jax.tree_map(
                lambda *x: jnp.concatenate(x), met_smpls, -met_smpls)
        )
        return samples, met_smpls

    def _nonlinear_sampling(self, samples):
        primals = samples.pos
        lh_trafo_at_p = self._lh_trafo(primals)
        metric_samples = self._linear_sampling(primals, False)[1]
        new_smpls = []
        opt_states = []
        for s, ms in zip(samples, metric_samples):
            options = {
                "fun_and_grad":
                    partial(
                        self._nl_vag,
                        p=primals,
                        lh_trafo_at_p=lh_trafo_at_p,
                        ms_at_p=ms
                    ),
                "hessp":
                    partial(
                        self._nl_metric,
                        p=primals,
                        lh_trafo_at_p=lh_trafo_at_p
                    ),
                "custom_gradnorm" : partial(self._nl_sampnorm, p=primals),
                }
            opt_state = minimize(None, x0=s, method=self._sampling_minimizer,
                                 options=self._sampling_kwargs | options)
            new_smpls.append(opt_state.x - primals)
            # Remove x from state to avoid copy of the samples
            opt_states.append(opt_state._replace(x = None))

        samples = Samples(pos=primals, samples=stack(new_smpls))
        return samples, opt_states

    def _minimize_kl(self, samples):
        options = {
            "fun_and_grad": partial(self._kl_vg, primals_samples=samples),
            "hessp": partial(self._kl_metric, primals_samples=samples),
        }
        opt_state = minimize(
            None,
            samples.pos,
            method=self._minimizer,
            options=self._mini_kwargs | options
        )
        return samples.at(opt_state.x), opt_state

    def init_state(self, primals):
        if self._sampling_method in ['linear', 'geometric']:
            smpls = self._linear_sampling(primals, False)[1]
        else:
            smpls = self._linear_sampling(primals, True)[0]
        state = OptVIState(niter=0, samples=smpls, sampling_states=None,
                           minimization_state=None)
        return primals, state

    def update(self, primals, state):
        if self._sampling_method in ['linear', 'geometric']:
            samples = self._linear_sampling(primals, True)[0]
        else:
            samples = state.samples.at(primals)

        if self._sampling_method in ['geometric', 'altmetric']:
            samples, sampling_states = self._nonlinear_sampling(samples)
        else:
            sampling_states = None
        samples, opt_state = self._minimize_kl(samples)
        state = OptVIState(niter=state.niter+1, samples=samples,
                           sampling_states=sampling_states,
                           minimization_state=opt_state)
        return samples.pos, state

    def run(self, primals):
        primals, state = self.init_state(primals)
        for i in range(self._n_iter):
            print(f"OptVI iteration number: {i}", file=sys.stderr)
            primals, state = self.update(primals, state)
        return primals, state

def _make_callable(obj):
    if isinstance(obj, dict):
        return {kk:_make_callable(ii) for kk, ii in obj.items()}
    if callable(obj) and not isinstance(obj, Likelihood):
        return obj
    else:
        return lambda x: obj

def _getitem(cfg, i):
    if not isinstance(cfg, dict):
        return cfg(i)
    return {kk: _getitem(ii,i) for kk,ii in cfg.items()}

def optimize_kl(
    likelihood: Likelihood,
    pos: Vector,
    total_iterations: int,
    n_samples: Union[int, Callable],
    key: jax.random.PRNGKey,
    minimizer: Union[str, Callable] = 'newtoncg',
    minimization_kwargs: dict = {},
    sampling_method: Union[str, Callable] = 'altmetric',
    sampling_minimizer: Union[str, Callable] = 'newtoncg',
    sampling_kwargs: dict = {'xtol':0.01},
    sampling_cg_kwargs: dict = {'maxiter':50},
    resample: Union[bool, Callable] = False,
    callback=None,
    out_dir=None,
    resume=False,
    verbosity=0):
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
    minimizer: str or callable
        Minimization method used for KL minimization.
    minimization_kwargs : dict
        Keyword arguments for minimizer used for KL minimization. Can also
        contain callables as entries in the dict, to change the parameters as a
        function of the current iteration.
    sampling_method: str or callable
        Sampling method used for vi approximation. Default is `altmetric`.
    sampling_minimizer: str or callable
        Minimization method used for non-linear sample minimization.
    sampling_kwargs: dict
        Keyword arguments for minimizer used for sample minimization. Can also
        contain callables as entries in the dict.
    sampling_cg_kwargs: dict
        Keyword arguments for ConjugateGradient used for the linear part of
        sample minimization. Can also contain callables as entries in the dict.
    resample: bool or callable
        Whether to resample with new random numbers or not. Default is False
    callback : callable or None
        Function that is called after every global iteration. It needs to be a
        function taking 3 arguments: 1. the position in latend space,
                                     2. the residual samples,
                                     3. the global iteration.
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
    """

    # Prepare dir and load last iteration
    if not out_dir == None:
        makedirs(out_dir, exist_ok=True)
    lfile = f"{out_dir}/last_finished_iteration"
    last_finished_index = -1
    if resume and isfile(lfile):
        with open(lfile) as f:
            last_finished_index = int(f.read())

    # Setup verbosity level
    sampling_cg_kwargs['name'] = None
    sampling_kwargs['name'] = None
    minimization_kwargs["name"] = None
    if verbosity >= 0:
        sampling_cg_kwargs['name'] = "linear_sampling"
        sampling_kwargs['name'] = "non_linear_sampling"
        minimization_kwargs["name"] = "minimize"
    if verbosity < 1:
        if "cg_kwargs" in minimization_kwargs.keys():
            minimization_kwargs["cg_kwargs"]["name"] = None
        else:
            minimization_kwargs["cg_kwargs"] = {"name": None}
        if "cg_kwargs" in sampling_kwargs.keys():
            sampling_kwargs["cg_kwargs"]["name"] = None
        else:
            sampling_kwargs["cg_kwargs"] = {"name": None}

    # Turn everything into callables by iteration number
    cfg = {'likelihood': likelihood,
           'n_samples': n_samples,
           'resample': resample,
           'sampling_method': sampling_method,
           'sampling_minimizer':  sampling_minimizer,
           'sampling_kwargs': sampling_kwargs,
           'sampling_cg_kwargs': sampling_cg_kwargs,
           'minimizer': minimizer,
           'minimization_kwargs': minimization_kwargs}
    cfg = {kk: _make_callable(ii) for kk,ii in cfg.items()}

    def get_optvi(n, key, _func = None):
        ncfg = _getitem(cfg, n)
        lh = ncfg['likelihood'] if _func is None else None
        opt = OptimizeVI(lh, 0, key,
                         ncfg['n_samples'],
                         ncfg['sampling_method'],
                         ncfg['sampling_minimizer'],
                         ncfg['sampling_kwargs'],
                         ncfg['sampling_cg_kwargs'],
                         ncfg['minimizer'],
                         ncfg['minimization_kwargs'],
                         _lh_funcs = _func)
        if (n > 0) and (lh is not None):
            msg = f"Warning: OptVI re-jit triggered at iteration number: {i}"
            print(msg, file=sys.stderr)
        return opt

    # Load last finished reconstruction
    if last_finished_index > -1:
        pos = pickle.load(
            open(f"{out_dir}/position_it_{last_finished_index}.p", "rb"))
        key = pickle.load(
            open(f"{out_dir}/rnd_key{last_finished_index}.p", "rb"))
        samples = pickle.load(
            open(f"{out_dir}/samples_{last_finished_index}.p", "rb"))
        state = OptVIState(niter=last_finished_index,
                           samples=samples,
                           sampling_states=None,
                           minimization_state=None)
        kp, sub = jax.random.split(key, 2)
        opt = get_optvi(last_finished_index+1, sub)
        nsam = _getitem(cfg['n_samples'], last_finished_index+1)
        onsam = _getitem(cfg['n_samples'], last_finished_index)
        do_resampling = (_getitem(cfg['resample'], last_finished_index+1) or
                         (nsam != onsam))
    else:
        pos = pos.copy()
        kp, sub = jax.random.split(key, 2)
        opt = get_optvi(0, sub)
        do_resampling = True

    for i in range(last_finished_index + 1, total_iterations):
        # Potentially re-initialize samples
        if do_resampling:
            pos, state = opt.init_state(pos)
        # Do one sampling and minimization step
        pos, state = opt.update(pos, state)
        en = state.minimization_state.fun
        print(f"Post VI Iteration {i}: Energy {en:2.4e}", file=sys.stderr)
        if state.sampling_states is not None:
            niter = tuple(ss.nit for ss in state.sampling_states)
            msg = f"Nonlinear sampling total iterations: {niter}"
            print(msg, file=sys.stderr)
        msg = f"KL-Minimization total iteration: {state.minimization_state.nit}"
        print(msg, file=sys.stderr)

        if callback != None:
            callback(pos, state, i)
        do_resampling = (_getitem(cfg['resample'], i+1) or
            (_getitem(cfg['n_samples'], i+1) != _getitem(cfg['n_samples'], i)))
        if do_resampling:
            key = kp
            kp, sub = jax.random.split(key)
        if i != total_iterations - 1:
            _func = opt.lh_funcs
            # If likelihood or sampling cg changes, trigger re-compilation
            if (_getitem(cfg['likelihood'], i+1) !=
                _getitem(cfg['likelihood'], i)):
                _func = None
            kwa = cfg['sampling_cg_kwargs']
            if _getitem(kwa, i+1) != _getitem(kwa, i):
                #TODO changing the linear sampling params triggers a full re-jit
                # of all functions (including those used for minimization).
                # This is not necessary and may cause unwanted overhead!
                _func = None
            opt = get_optvi(i+1, sub, _func = _func)

        if not out_dir == None:
            # Save iteration
            pickle.dump(pos, open(f"{out_dir}/position_it_{i}.p", "wb"))
            pickle.dump(state.samples, open(f"{out_dir}/samples_{i}.p", "wb"))
            pickle.dump(key, open(f"{out_dir}/rnd_key{i}.p", "wb"))
            with open(f"{out_dir}/last_finished_iteration", "w") as f:
                f.write(str(i))

    return pos, state.samples