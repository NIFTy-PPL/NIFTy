# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Author: Jakob Roth, Laurin Söding

import re
import math
import jax
from timeit import Timer

from jax.tree_util import Partial

from .model import LazyModel
from .logger import logger
from .tree_math import ones_like


def _benchmark(func, *args, **kwargs):
    f = lambda: jax.block_until_ready(func(*args, **kwargs))
    _ = f()  # warmup
    t = Timer(f)
    n, delta_t = t.autorange()
    delta_t /= n
    return delta_t


def _dtype_to_bits(dtype):
    match = re.search(r"(\d+)$", str(dtype))
    return int(match.group(1)) if match else float("nan")


def _parse_hlo(hlo):
    # matches lines like "%constant.123 = f32[10, 20]""
    pattern = r"^\s*%constant[\.\d]*\s*=\s*([a-zA-Z0-9]+)\[([0-9,\s]*)\]"
    matches = re.findall(pattern, hlo, re.MULTILINE)
    constants_shapes = {}
    for dtype, shape_str in matches:
        if shape_str.strip() == "":
            shape = []  # scalar
        else:
            shape = [int(x.strip()) for x in shape_str.split(",")]
        # results.append((dtype, shape))
        prev_consts = constants_shapes.get(dtype, [])
        prev_consts.append(shape)
        constants_shapes[dtype] = prev_consts

    total_size = {}
    memory_size = {}
    for dtype, shapes in constants_shapes.items():
        constants_shapes[dtype] = sorted(
            shapes, key=lambda s: math.prod(s) if s else 0, reverse=True
        )
        total_size[dtype] = sum(math.prod(s) for s in shapes)
        memory_size[dtype] = _dtype_to_bits(dtype) * total_size[dtype] / 8
    return constants_shapes, total_size, memory_size


def _log_hlo_consts(name, consts, size, mem_bytes):
    msg = f"\nHLO parsing {name}:\n"
    for dtype in consts.keys():
        msg += (
            f"  * constants of type {dtype}:\n"
            f"      - shapes of 5 largest constants: {consts[dtype][:5]}\n"
            f"      - total size of constants: {size[dtype]}\n"
            f"      - total memory of constants: {mem_bytes[dtype]:.1e} bytes\n"
        )
    logger.info(msg)


def _log_model_leaves(model):
    leaves = jax.tree.leaves(model)
    msg = "\n leaves of model:\n"
    for l in leaves:
        if isinstance(l, jax.Array):
            msg += f"  * shape: {l.shape} dtype: {l.dtype}\n"
        else:
            msg += f"  * leaf of non-Array type {type(l)}\n"
    logger.info(msg)


def check_model(model, pos):
    """
    Benchmarks and analyzes a NIFTy model's forward pass, JVP, and VJP.

    Runs four types of analysis for each of the three evaluation modes
    (forward, JVP, VJP):

    - **Timing:** Measures execution time with and without JIT compilation.
    - **Memory analysis:** Reports memory usage of the JIT-compiled computations.
    - **HLO parsing:** Inspects the compiled HLO representation to analyze
      inlined constants.
    - **Model leaves:** Inspects the leaves of the model which are not inlined but
      treated as variables.

    Parameters
    ----------
    model: :class:`~nifty.re.model.Model` or callable
        The NIFTy Model to be benchmarked and analyzed.
    pos : Any
        Input to the model. Must be a JAX-compatible pytree of arrays.
    """
    model = model if isinstance(model, LazyModel) else Partial(model)
    cotangent = ones_like(jax.eval_shape(model, pos))

    modes = {
        "forward": (lambda m, x: m(x), (model, pos)),
        "jvp": (lambda m, p, t: jax.jvp(m, [p], [t]), (model, pos, pos)),
        "vjp": (lambda m, p, t: jax.vjp(m, p)[1](t), (model, pos, cotangent)),
    }

    for name, (fn, args) in modes.items():
        compiled = jax.jit(fn).lower(*args).compile()

        time_raw = _benchmark(fn, *args)
        time_jit = _benchmark(compiled, *args)
        mem = compiled.memory_analysis()
        hlo = compiled.as_text()
        consts, sizes, mem_bytes = _parse_hlo(hlo)

        logger.info(
            f"=== {name} ===\n"
            f"  * time (no jit): {time_raw:.1e}s\n"
            f"  * time (jit):    {time_jit:.1e}s\n"
            f"  * memory:\n{str(mem)}\n"
        )
        _log_hlo_consts(name, consts, sizes, mem_bytes)

    _log_model_leaves(model)
