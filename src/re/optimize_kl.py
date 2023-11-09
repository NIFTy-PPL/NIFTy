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
from .logger import logger


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
    logger.info(f"Post VI Iteration {iiter}: Energy {en:2.4e}")
    if state.sampling_states is not None:
        niter = tuple(ss.nit for ss in state.sampling_states)
        logger.info(f"Nonlinear sampling total iterations: {niter}")

    msg = f"KL-Minimization total iteration: {state.minimization_state.nit}"
    logger.info(msg)
    _, minis = minisanity(primals, state.samples, residual)
    msg = "Likelihood residual(s):\n"
    msg += minis
    logger.info(msg)
    _, minis = minisanity(primals, state.samples)
    msg = "Prior residual(s):\n"
    msg += minis
    logger.info(msg)


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
    linear_sampling_kwargs: dict = {'cg_kwargs':{'maxiter':50}},
    sampling_minimizer: Union[str, Callable] = 'newtoncg',
    sampling_kwargs: dict = {'xtol':0.01},
    resample: Union[bool, Callable] = False,
    kl_kwargs: dict = {},
    curve_kwargs: dict = {},
    callback=None,
    out_dir=None,
    resume=False,
    verbosity=0):
    # TODO update docstring
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
    if verbosity < 0:
        linear_sampling_kwargs['cg_kwargs']['name'] = None
        sampling_kwargs['name'] = None
        minimization_kwargs['name'] = None
    else:
        linear_sampling_kwargs['cg_kwargs'].setdefault('name','linear_sampling')
        sampling_kwargs.setdefault('name', 'non_linear_sampling')
        minimization_kwargs.setdefault('name', 'minimize')
    if verbosity < 1:
        if "cg_kwargs" in minimization_kwargs.keys():
            minimization_kwargs["cg_kwargs"].set_default('name', None)
        else:
            minimization_kwargs["cg_kwargs"] = {"name": None}
        if "cg_kwargs" in sampling_kwargs.keys():
            sampling_kwargs["cg_kwargs"].set_default('name', None)
        else:
            sampling_kwargs["cg_kwargs"] = {"name": None}

    # Split into state changing inputs and constructor inputs of OptimizeVI
    state_cfg = {
        'n_samples': n_samples,
        'sampling_method': sampling_method,
        'resample': resample,
    }
    update_cfg = {
        'kl_minimizer': minimizer,
        'kl_minimizer_kwargs': minimization_kwargs,
        'curve_minimizer': sampling_minimizer,
        'curve_minimizer_kwargs': sampling_kwargs,
    }

    constructor_cfg = {
        'likelihood': likelihood,
        'linear_sampling_kwargs': linear_sampling_kwargs,
        'point_estimates': point_estimates,
        'kl_kwargs': kl_kwargs,
        'curve_kwargs': curve_kwargs,
    }
    # Turn everything into callables by iteration number
    state_cfg = {kk: _make_callable(ii) for kk,ii in state_cfg.items()}
    update_cfg = {kk: _make_callable(ii) for kk,ii in update_cfg.items()}
    constructor_cfg = {kk: _make_callable(ii) for kk,ii in
                       constructor_cfg.items()}

    # Initialize Optimizer
    opt = OptimizeVI(n_iter=total_iterations,
                     **_getitem(constructor_cfg, last_finished_index+1))

    # Load last finished reconstruction
    if last_finished_index > -1:
        pos = pickle.load(
            open(f"{out_dir}/position_{last_finished_index}.p", "rb"))
        key = pickle.load(
            open(f"{out_dir}/rnd_key_{last_finished_index}.p", "rb"))
        state = pickle.load(
            open(f"{out_dir}/state_{last_finished_index}.p", "rb"))
        if last_finished_index == total_iterations - 1:
            return pos, state
    else:
        keys = jax.random.split(key, _getitem(state_cfg['n_samples'], 0)+1)
        key = keys[0]
        state = opt.init_state(pos, keys[1:],
            sampling_method=_getitem(state_cfg['sampling_method'], 0)
        )

    # Update loop
    for i in range(last_finished_index + 1, total_iterations):
        # Do one sampling and minimization step
        pos, state = opt.update(pos, state, **_getitem(update_cfg, i))
        # Print basic infos (TODO: save to ouput file)
        basic_status_print(i, pos, state, likelihood.normalized_residual)
        if callback != None:
            callback(pos, state, i)

        if i != total_iterations - 1:
            # Update state
            do_resampling = (_getitem(state_cfg['resample'], i+1) or
                                (_getitem(state_cfg['n_samples'], i+1) !=
                                _getitem(state_cfg['n_samples'], i))
                            )
            if do_resampling:
                keys = jax.random.split(key, _getitem(state_cfg['n_samples'], 0)+1)
                key = keys[0]
                state = state._replace(keys=keys[1:])
            state = state._replace(
                resample=do_resampling,
                sampling_method=_getitem(state_cfg['sampling_method'], i+1)
                )

            # Check for update in constructor and re-initialize sampler
            rebuild = False
            for rr in constructor_cfg.keys():
                if (_getitem(constructor_cfg[rr], i+1) !=
                    _getitem(constructor_cfg[rr], i)):
                    rebuild = True
            if rebuild:
                # TODO print warning
                # TODO only partial rebuild
                opt = OptimizeVI(n_iter=total_iterations,
                                 **_getitem(constructor_cfg, i+1))

        if not out_dir == None:
            # TODO: Make this fail safe! Cancelling the run while partially
            # saving the outputs may result in a corrupted state.
            # Save iteration
            pickle.dump(key, open(f"{out_dir}/rnd_key_{i}.p", "wb"))
            pickle.dump(pos, open(f"{out_dir}/position_{i}.p", "wb"))
            pickle.dump(state, open(f"{out_dir}/state_{i}.p", "wb"))
            with open(f"{out_dir}/last_finished_iteration", "w") as f:
                f.write(str(i))

    return pos, state