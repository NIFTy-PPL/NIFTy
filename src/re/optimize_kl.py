#!/usr/bin/env python3
# Copyright(C) 2013-2023 Max-Planck-Society
# SPDX-License-Identifier: BSD-2-Clause
# Authors: Philipp Frank, Jakob Roth

from functools import reduce
import pickle
import jax
from os import makedirs
from os.path import isfile
from typing import Callable, Union, Tuple
from .tree_math.vector import Vector
from .likelihood import Likelihood
from .kl import OptVIState, OptimizeVI, optimizeVI_callables
from .misc import minisanity
from .logger import logger


def _make_callable(obj):
    if callable(obj) and not isinstance(obj, Likelihood):
        return obj
    else:
        return lambda x: obj


def _getitem(cfg, i):
    if not isinstance(cfg, dict):
        return cfg(i)
    return {kk: _getitem(ii,i) for kk,ii in cfg.items()}


def basic_status_print(iiter, samples, state, residual, out_dir=None):
    en = state.minimization_state.fun
    msg = f"Post VI Iteration {iiter}: Energy {en:2.4e}\n"
    if state.sampling_states is not None:
        niter = tuple(ss.nit for ss in state.sampling_states)
        msg += f"Nonlinear sampling total iterations: {niter}\n"
    msg += f"KL-Minimization total iteration: {state.minimization_state.nit}\n"
    _, minis = minisanity(samples.pos, samples, residual)
    msg += "Likelihood residual(s):\n"
    msg += minis +"\n"
    _, minis = minisanity(samples.pos, samples)
    msg += "Prior residual(s):\n"
    msg += minis+"\n"
    logger.info(msg)

    if not out_dir == None:
        lfile = f"{out_dir}/minisanity"
        if isfile(lfile) and iiter != 0:
            with open(lfile) as f:
                msg = str(f.read()) + "\n" + msg
        with open(f"{out_dir}/minisanity", "w") as f:
            f.write(msg)


def _do_resample(cfg, iiter):
    if iiter == 0:
        return True
    cfgi = _getitem(cfg, iiter)
    cfgo = _getitem(cfg, iiter-1)
    regenerate = cfgi['resample']
    regenerate += (cfgi['n_samples'] != cfgo['n_samples'])
    return bool(regenerate)


def update_state(state, cfg, iiter):
    # This configures the generic interface of `OptimizeVI` for the specific
    # cases of the `linear`, `geometric`, `altmetric` methods.
    regenerate = (_getitem(cfg, iiter)['sampling_method'] in
                  ['linear', 'geometric'])
    update = (_getitem(cfg, iiter)['sampling_method'] in
              ['geometric', 'altmetric'])
    state = state._replace(
        sample_regenerate=regenerate or _do_resample(cfg, iiter),
        sample_update=update,
        kl_solver_kwargs=_getitem(cfg, iiter)['kl_solver_kwargs'],
        sample_generator_kwargs=_getitem(cfg, iiter)['sample_generator_kwargs'],
        sample_update_kwargs=_getitem(cfg, iiter)['sample_update_kwargs'],
    )
    return state


def optimize_kl(
    likelihood: Union[Likelihood, Callable, None],
    pos: Vector,
    total_iterations: int,
    n_samples: Union[int, Callable],
    key: jax.random.PRNGKey,
    point_estimates: Union[Vector, Tuple[str], Callable] = (),
    sampling_method: Union[str, Callable] = 'altmetric',
    make_kl_kwargs: Union[dict, Callable] = {},
    make_sample_generator_kwargs: Union[dict, Callable] = {
        'cg_kwargs':{'maxiter':50}
    },
    make_sample_update_kwargs: Union[dict, Callable] = {},
    kl_solver_kwargs: Union[dict, Callable] = {
        'method': 'newtoncg',
        'method_options': {},
    },
    sample_generator_kwargs: dict = {},
    sample_update_kwargs: dict = {
        'method': 'newtoncg',
        'method_options': {'xtol':0.01},
    },
    resample: Union[bool, Callable] = False,
    callback=None,
    out_dir=None,
    resume=False,
    verbosity=0,
    _vi_callables: Union[None, Tuple[Callable], Callable] = None,
    _update_state: Callable = update_state):
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
        make_sample_generator_kwargs['cg_kwargs']['name'] = None
        kl_solver_kwargs['method_options']['name'] = None
        sample_update_kwargs['method_options']['name'] = None
    else:
        make_sample_generator_kwargs['cg_kwargs'].setdefault(
            'name', 'linear_sampling'
        )
        sample_update_kwargs['method_options'].setdefault(
            'name', 'non_linear_sampling'
        )
        kl_solver_kwargs['method_options'].setdefault(
            'name', 'minimize'
        )
    if verbosity < 1:
        if "cg_kwargs" in kl_solver_kwargs['method_options'].keys():
            kl_solver_kwargs['method_options']["cg_kwargs"].set_default(
                'name', None
            )
        else:
            kl_solver_kwargs['method_options']["cg_kwargs"] = {"name": None}
        if "cg_kwargs" in sample_update_kwargs['method_options'].keys():
            sample_update_kwargs['method_options']["cg_kwargs"].set_default(
                'name', None
            )
        else:
            sample_update_kwargs['method_options']["cg_kwargs"] = {"name": None}

    # Split into state changing inputs and constructor inputs of OptimizeVI
    state_cfg = {
        'n_samples': n_samples,
        'sampling_method': sampling_method,
        'resample': resample,
        'kl_solver_kwargs': kl_solver_kwargs,
        'sample_generator_kwargs':sample_generator_kwargs,
        'sample_update_kwargs':sample_update_kwargs,
    }
    constructor_cfg = {
        'likelihood': likelihood,
        'point_estimates': point_estimates,
        'linear_sampling_kwargs': make_sample_generator_kwargs,
        'kl_kwargs': make_kl_kwargs,
        'curve_kwargs': make_sample_update_kwargs,
    }
    # Turn everything into callables by iteration number
    state_cfg = {kk: _make_callable(ii) for kk,ii in state_cfg.items()}
    constructor_cfg = {kk: _make_callable(ii) for kk,ii in
                       constructor_cfg.items()}
    _vi_callables = _make_callable(_vi_callables)

    # Initialize Optimizer
    # If `_vi_callables` are set, use them to set up optimizer directly instead
    # of using `optimizeVI_callables`
    vic = _getitem(_vi_callables, last_finished_index+1)
    if vic is not None:
        opt = OptimizeVI(n_iter=total_iterations, *vic)
    else:
        kl, lin, geo = optimizeVI_callables(
            **_getitem(constructor_cfg, last_finished_index+1)
        )
        opt = OptimizeVI(n_iter=total_iterations,
                         kl_solver=kl,
                         sample_generator=lin,
                         sample_update=geo)

    # Load last finished reconstruction
    if last_finished_index > -1:
        samples = pickle.load(
            open(f"{out_dir}/samples_{last_finished_index}.p", "rb"))
        key = pickle.load(
            open(f"{out_dir}/rnd_key_{last_finished_index}.p", "rb"))
        state = pickle.load(
            open(f"{out_dir}/state_{last_finished_index}.p", "rb"))
        if last_finished_index == total_iterations - 1:
            return samples, state
    else:
        keys = jax.random.split(key, _getitem(state_cfg['n_samples'], 0)+1)
        key = keys[0]
        samples, state = opt.init_state(keys[1:], primals=pos)
        state = _update_state(state, state_cfg, 0)

    # Update loop
    for i in range(last_finished_index + 1, total_iterations):
        # Do one sampling and minimization step
        samples, state = opt.update(samples, state)
        # Print basic infos
        basic_status_print(i, samples, state, likelihood.normalized_residual,
                           out_dir=out_dir)
        if callback != None:
            callback(samples, state, i)

        if i != total_iterations - 1:
            # Update state
            state = _update_state(state, state_cfg, i+1)
            if _do_resample(state_cfg, i+1):
                # Update keys
                keys = jax.random.split(key,
                                        _getitem(state_cfg['n_samples'],0) + 1)
                key = keys[0]
                state = state._replace(keys=keys[1:])

            # Check for update in constructor and re-initialize sampler
            vic = _getitem(_vi_callables, i)
            if vic is not None:
                if vic != _getitem(_vi_callables, i+1):
                    opt.set_kl_solver(vic[0])
                    opt.set_sample_generator(vic[1])
                    opt.set_sample_update(vic[2])
            else:
                keep = reduce(lambda a,b: a*b,
                    (_getitem(constructor_cfg[rr], i+1) ==
                     _getitem(constructor_cfg[rr], i) for rr in
                     constructor_cfg.keys()),
                    True
                )
                if not keep:
                    # TODO print warning
                    # TODO only partial rebuild
                    funcs = optimizeVI_callables(
                        n_iter=total_iterations,
                        **_getitem(constructor_cfg, i+1)
                    )
                    opt.set_kl_solver(funcs[0])
                    opt.set_sample_generator(funcs[1])
                    opt.set_sample_update(funcs[2])

        if not out_dir == None:
            # TODO: Make this fail safe! Cancelling the run while partially
            # saving the outputs may result in a corrupted state.
            # Save iteration
            pickle.dump(key, open(f"{out_dir}/rnd_key_{i}.p", "wb"))
            pickle.dump(samples, open(f"{out_dir}/samples_{i}.p", "wb"))
            pickle.dump(state, open(f"{out_dir}/state_{i}.p", "wb"))
            with open(f"{out_dir}/last_finished_iteration", "w") as f:
                f.write(str(i))

    return samples, state