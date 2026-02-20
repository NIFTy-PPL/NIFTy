# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Author: Jakob Roth

import re
import math
import jax
import time

from jax.tree_util import Partial

from .model import LazyModel
from .logger import logger


def _benchmark(func, *args, **kwargs):
    start = time.perf_counter()
    _ = func(*args, **kwargs)
    delta_t = time.perf_counter() - start
    return delta_t


def _dtype_to_bits(dtype):
    match = re.search(r"(\d+)$", str(dtype))
    return int(match.group(1)) if match else float("nan")


def _parse_hlo(hlo):
    pattern = r"^\s*%constant\.\d+\s*=\s*([a-zA-Z0-9]+)\[([0-9,\s]*)\]"
    matches = re.findall(pattern, hlo, re.MULTILINE)
    results = {}
    for dtype, shape_str in matches:
        if shape_str.strip() == "":
            shape = []  # scalar
        else:
            shape = [int(x.strip()) for x in shape_str.split(",")]
        # results.append((dtype, shape))
        prev_consts = results.get(dtype, [])
        prev_consts.append(shape)
        results[dtype] = prev_consts

    total_size = {}
    memory_size = {}
    for dtype, shapes in results.items():
        results[dtype] = sorted(
            shapes, key=lambda s: math.prod(s) if s else 0, reverse=True
        )
        total_size[dtype] = sum(math.prod(s) for s in shapes)
        memory_size[dtype] = _dtype_to_bits(dtype) * total_size[dtype] / 8
    return results, total_size, memory_size


def _logg_hlo_consts(consts, size, bytes):
    for dtype in consts.keys():
        logger.info(f"constants of type {dtype}")
        logger.info(f"5 largest constants: {consts[dtype][:5]}")
        logger.info(f"total size of constants: {size[dtype]}")
        logger.info(f"total memory of constants: {bytes[dtype]:.1e} bytes\n")


def check_model(model, pos):
    """
    Benchmarks and analyzes a NIFTy model's forward pass, JVP, and VJP.

    Runs three types of analysis for each of the three evaluation modes
    (forward, JVP, VJP):

    - **Timing:** Measures execution time with and without JIT compilation.
    - **Memory analysis:** Reports memory usage of the JIT-compiled computations.
    - **HLO parsing:** Inspects the compiled HLO representation to analyze
      inlined constants.


    Parameters
    ----------
    model: :class:`~nifty.re.model.Model` or callable
        The NIFTy Model to be benchmarked and analyzed.
    pos : Any
        Input to the model. Must be a JAX-compatible pytree of arrays.
    """
    model = model if isinstance(model, LazyModel) else Partial(model)
    m_forward = lambda m, x: m(x)
    m_jvp = lambda m, p, t: jax.jvp(m, [p], [t])
    m_vjp = lambda m, p, t: jax.vjp(m, p)[1](t)

    m_forward_jit = jax.jit(m_forward)
    m_jvp_jit = jax.jit(m_jvp)
    m_vjp_jit = jax.jit(m_vjp)

    res_forward = m_forward_jit(model, pos)
    res_jvp = m_jvp_jit(model, pos, pos)
    res_vjp = m_vjp_jit(model, pos, res_forward)

    time_forward = _benchmark(m_forward, model, pos)
    time_jvp = _benchmark(m_jvp, model, pos, pos)
    time_vjp = _benchmark(m_vjp, model, pos, res_forward)
    logger.info(f"time forward: {time_forward:.1e} seconds")
    logger.info(f"time jvp: {time_jvp:.1e} seconds")
    logger.info(f"time vjp: {time_vjp:.1e} seconds\n")

    time_forward_jit = _benchmark(m_forward_jit, model, pos)
    time_jvp_jit = _benchmark(m_jvp_jit, model, pos, pos)
    time_vjp_jit = _benchmark(m_vjp_jit, model, pos, res_forward)
    logger.info(f"time forward jit: {time_forward_jit:.1e} seconds")
    logger.info(f"time jvp jit: {time_jvp_jit:.1e} seconds")
    logger.info(f"time vjp jit: {time_vjp_jit:.1e} seconds\n")

    mem_forward = m_forward_jit.lower(model, pos).compile().memory_analysis()
    mem_jvp = m_jvp_jit.lower(model, pos, pos).compile().memory_analysis()
    mem_vjp = m_vjp_jit.lower(model, pos, res_forward).compile().memory_analysis()

    msg_forward = "memory analysis forward:\n" + str(mem_forward) + "\n"
    msg_jvp = "memory analysis jvp:\n" + str(mem_jvp) + "\n"
    msg_vjp = "memory analysis vjp:\n" + str(mem_vjp) + "\n"
    logger.info(msg_forward)
    logger.info(msg_jvp)
    logger.info(msg_vjp)

    hlo_forward = m_forward_jit.lower(model, pos).compile().as_text()
    hlo_jvp = m_jvp_jit.lower(model, pos, pos).compile().as_text()
    hlo_vjp = m_vjp_jit.lower(model, pos, res_forward).compile().as_text()

    const_forward, size_forward, bytes_forward = _parse_hlo(hlo_forward)
    const_jvp, size_jvp, bytes_jvp = _parse_hlo(hlo_jvp)
    const_vjp, size_vjp, bytes_vjp = _parse_hlo(hlo_vjp)

    logger.info("hlo parsing forward:")
    _logg_hlo_consts(const_forward, size_forward, bytes_forward)

    logger.info("hlo parsing jvp:")
    _logg_hlo_consts(const_jvp, size_jvp, bytes_jvp)

    logger.info("hlo parsing vjp:")
    _logg_hlo_consts(const_vjp, size_vjp, bytes_vjp)
