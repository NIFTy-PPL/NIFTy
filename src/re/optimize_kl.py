#!/usr/bin/env python3
# Copyright(C) 2013-2023 Max-Planck-Society
# SPDX-License-Identifier: BSD-2-Clause
# Authors: Philipp Frank, Jakob Roth

import pickle
import sys
import jax
from os import makedirs
from os.path import isfile
from typing import Callable, Union, Tuple
from .tree_math.vector import Vector
from .likelihood import Likelihood
from .kl import OptimizeVI, OptVIState
from .misc import minisanity


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


def basic_status_print(iiter, primals, state, residual):
    en = state.minimization_state.fun
    print(f"Post VI Iteration {iiter}: Energy {en:2.4e}", file=sys.stderr)
    if state.sampling_states is not None:
        niter = tuple(ss.nit for ss in state.sampling_states)
        msg = f"Nonlinear sampling total iterations: {niter}"
        print(msg, file=sys.stderr)
    msg = f"KL-Minimization total iteration: {state.minimization_state.nit}"
    print(msg, file=sys.stderr)
    _, minis = minisanity(primals, state.samples, residual)
    print("Likelihood residual(s):", file=sys.stderr)
    print(minis, file=sys.stderr)
    _, minis = minisanity(primals, state.samples)
    print("Prior residual(s):", file=sys.stderr)
    print(minis, file=sys.stderr)


def optimize_kl(
    likelihood: Likelihood,
    pos: Vector,
    total_iterations: int,
    n_samples: Union[int, Callable],
    key: jax.random.PRNGKey,
    point_estimates: Union[Vector, Tuple[str]] = (),
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
    point_estimates : tree-like structure or tuple of str
        Pytree of same structure as `pos` but with boolean leaves indicating
        whether to sample the value in `pos` or use it as a point estimate. As
        a convenience method, for dict-like `pos`, a tuple of strings is also
        valid. From these the boolean indicator pytree is automatically
        constructed.
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
           'point_estimates': point_estimates,
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
                         ncfg['point_estimates'],
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
        # Print basic infos (TODO: save to ouput file)
        basic_status_print(i, pos, state, likelihood.normalized_residual)

        if callback != None:
            callback(pos, state, i)
        do_resampling = (_getitem(cfg['resample'], i+1) or
            (_getitem(cfg['n_samples'], i+1) != _getitem(cfg['n_samples'], i)))
        if do_resampling:
            key = kp
            kp, sub = jax.random.split(key)
        if i != total_iterations - 1:
            _func = opt.lh_funcs
            # If likelihood, point_estimates, or sampling_cg changes, trigger
            # re-compilation.
            #TODO changing sampling_cg triggers a full re-jit of all functions
            # (including those used for minimization). This is not necessary and
            # may cause unwanted overhead!
            re_jit = ['likelihood', 'point_estimates', 'sampling_cg_kwargs']
            for rr in re_jit:
                if (_getitem(cfg[rr], i+1) != _getitem(cfg[rr], i)):
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