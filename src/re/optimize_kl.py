# Copyright(C) 2023 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import jax
import sys
import pickle
import jax.numpy as jnp

from functools import partial
from jax import random
from os import makedirs
from os.path import isfile

from .likelihood import StandardHamiltonian
from .kl import Samples, sample_evi
from .smap import smap
from .tree_math import Vector, stack
from .optimize import minimize


def _make_callable(obj):
    if callable(obj) and not isinstance(obj, int):
        return obj
    else:
        return lambda x: obj


def optimize_kl(
    likelihood,
    pos,
    total_iterations,
    n_samples,
    newton_cg_kwargs,
    linear_sampling_kwargs,
    non_linear_sampling_kwargs,
    key,
    callback=None,
    out_dir=None,
    resume=False,
    verbosity=0,
    reuse_rnd_numbers=False,
):
    """
    Interface for KL minimization similar to NIFTy optimize_kl.


    Parameters
    ----------
    likelihood : :class:`nifty8.re.likelihood.Likelihood`
        Likelihood to be used for inference.
    pos : Initial position for minimization.
    total_iterations : int
        Number of resampling loops.
    n_samples : int or callable
        Number of samples used to sample Kullback-Leibler divergence.
    newton_cg_kwargs : dict
        keyword arguments for NewtonCG used for KL minimization.
    linear_sampling_kwargs : dict
        keyword arguments used for linear sampling.
    non_linear_sampling_kwargs : dict or None
        keyword arguments used for non linear sampling. Only linear sampling
        if set to None.
    key : jax random number generataion key
    callback : callable or None
        Function that is called after every global iteration. It need to be a function
        taking 3 arguments: 1. the position in latend space, 2. the residual samples,
        3. the global iteration. Default: None.
    output_directory : str or None
        Directory in which all output files are saved. If None, no output is
        stored.  Default: None.
    resume : bool
        Resume partially run optimization. If `True` and `output_directory`
        is specified it resumes optimization. Default: False.
    verbosity : int
        Sets verbosity of optimization. If -1 only the current global optimization
        index is printed. If 0 CG steps of linear sampling, NewtonCG steps of
        non linear sampling and NewtonCG steps of KL optimization are printed.
        If set to 1 additionally the internal CG steps of the NewtonCG optimization
        are printed. Default: 0.
    reuse_rnd_numbers : bool
        If set to True random numbers for drawing samples will be reused. This can minimize
        the stochasticity of the optimization. Default: False

    """
    n_samples = _make_callable(n_samples)
    if not out_dir == None:
        makedirs(out_dir, exist_ok=True)
    lfile = f"{out_dir}/last_finished_iteration"
    last_finished_index = -1
    if resume and isfile(lfile):
        with open(lfile) as f:
            last_finished_index = int(f.read())

    run_geoVI = False
    if not non_linear_sampling_kwargs == None:
        run_geoVI = True

    ls_name = None
    nls_name = None
    newton_cg_kwargs["name"] = None
    if verbosity >= 0:
        ls_name = "linear_sampling"
        nls_name = "non_linear_sampling"
        newton_cg_kwargs["name"] = "newton"
    if verbosity < 1:
        if "cg_kwargs" in newton_cg_kwargs.keys():
            newton_cg_kwargs["cg_kwargs"]["name"] = None
        else:
            newton_cg_kwargs["cg_kwargs"] = {"name": None}
        if not non_linear_sampling_kwargs is None:
            if "cg_kwargs" in non_linear_sampling_kwargs.keys():
                non_linear_sampling_kwargs["cg_kwargs"]["name"] = None
            else:
                non_linear_sampling_kwargs["cg_kwargs"] = {"name": None}

    likelihood = likelihood.jit()
    ham = StandardHamiltonian(likelihood=likelihood).jit()

    @jax.jit
    def ham_vg(primals, primals_samples):
        assert isinstance(primals_samples, Samples)
        vvg = jax.vmap(jax.value_and_grad(ham))
        s = vvg(primals_samples.at(primals).samples)
        return jax.tree_util.tree_map(partial(jnp.mean, axis=0), s)

    @jax.jit
    def ham_metric(primals, tangents, primals_samples):
        assert isinstance(primals_samples, Samples)
        vmet = jax.vmap(ham.metric, in_axes=(0, None))
        s = vmet(primals_samples.at(primals).samples, tangents)
        return jax.tree_util.tree_map(partial(jnp.mean, axis=0), s)

    @partial(jax.jit, static_argnames=("point_estimates",))
    def sample_mgvi(primals, key, *, point_estimates=()):
        # at: reset relative position as it gets (wrongly) batched too
        # squeeze: merge "samples" axis with "mirrored_samples" axis
        return (
            smap(
                partial(
                    sample_evi,
                    likelihood,
                    linear_sampling_name=ls_name,
                    linear_sampling_kwargs=linear_sampling_kwargs,
                    point_estimates=point_estimates,
                ),
                in_axes=(None, 0),
            )(primals, key)
            .at(primals)
            .squeeze()
        )

    def sample_geovi(primals, key, *, point_estimates=()):
        sample = partial(
            sample_evi,
            likelihood,
            linear_sampling_name=ls_name,
            non_linear_sampling_name=nls_name,
            linear_sampling_kwargs=linear_sampling_kwargs,
            point_estimates=point_estimates,
            non_linear_sampling_kwargs=non_linear_sampling_kwargs,
        )
        # Manually loop over the keys for the samples because mapping over them
        # would implicitly JIT the sampling which is exacty what we want to avoid.
        samples = tuple(sample(primals, k) for k in key)
        # at: reset relative position as it gets (wrongly) batched too
        # squeeze: merge "samples" axis with "mirrored_samples" axis
        return stack(samples).at(primals).squeeze()

    if last_finished_index > -1:
        pos = pickle.load(open(f"{out_dir}/position_it_{last_finished_index}.p", "rb"))
        key = pickle.load(open(f"{out_dir}/rnd_key{last_finished_index}.p", "rb"))
        if last_finished_index == total_iterations - 1:
            samples = pickle.load(
                open(f"{out_dir}/samples_{last_finished_index}.p", "rb")
            )
    else:
        pos = Vector(pos.copy())

    key, subkey = random.split(key, 2)
    for i in range(last_finished_index + 1, total_iterations):
        if run_geoVI:
            print(f"geoVI Iteration {i}", file=sys.stderr)
        else:
            print(f"MGVI Iteration {i}", file=sys.stderr)

        print("Sampling...", file=sys.stderr)

        if run_geoVI:
            samples = sample_geovi(pos, random.split(subkey, n_samples(i)))
        else:
            samples = sample_mgvi(pos, random.split(subkey, n_samples(i)))

        print("Minimizing...", file=sys.stderr)
        opt_state = minimize(
            None,
            pos,
            method="newton-cg",
            options={
                **{
                    "fun_and_grad": partial(ham_vg, primals_samples=samples),
                    "hessp": partial(ham_metric, primals_samples=samples),
                },
                **newton_cg_kwargs,
            },
        )
        pos = opt_state.x
        msg = f"Post MGVI Iteration {i}: Energy {ham_vg(pos, samples)[0]:2.4e}"
        print(msg, file=sys.stderr)
        if not callback == None:
            callback(pos, samples, i)
        if not out_dir == None:
            pickle.dump(pos, open(f"{out_dir}/position_it_{i}.p", "wb"))
            pickle.dump(samples, open(f"{out_dir}/samples_{i}.p", "wb"))
            pickle.dump(key, open(f"{out_dir}/rnd_key{i}.p", "wb"))
            with open(f"{out_dir}/last_finished_iteration", "w") as f:
                f.write(str(i))
        if not reuse_rnd_numbers:
            key, subkey = random.split(key, 2)

    return pos, samples, key
