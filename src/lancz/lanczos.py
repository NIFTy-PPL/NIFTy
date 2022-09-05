# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from functools import partial
from typing import Callable, Optional, Union
# from joblib import Parallel, delayed

import numpy as np
from operator import matmul

from numpy.random import PCG64, Generator, bit_generator


def lanczos_tridiag(
    mat: Callable, shape_dtype_tuple: tuple, order: int,
    seed: bit_generator.SeedSequence
):
    """Compute the Lanczos decomposition into a tri-diagonal matrix and its
    corresponding orthonormal projection matrix.
    """
    rng = np.random.default_rng(seed)
    tridiag = np.zeros((order, order), dtype=shape_dtype_tuple[1])
    vecs = np.zeros(
        (order, ) + shape_dtype_tuple[0], dtype=shape_dtype_tuple[1]
    )

    # v = random.normal(key, shape=shape_dtype_tuple[0])
    v = rng.random(size=shape_dtype_tuple[0])
    v = v / np.linalg.norm(v)
    vecs[0] = v

    # Zeroth iteration
    w = mat(v)
    w = np.copy(w) #FIXME
    if w.shape != shape_dtype_tuple[0]:
        ve = f"shape of `mat(v)` {w.shape!r} incompatible with {shape_dtype_tuple}"
        raise ValueError(ve)
    alpha = np.dot(w, v)
    tridiag[(0, 0)] = alpha
    w -= alpha * v
    beta = np.linalg.norm(w)

    tridiag[(0, 1)] = beta
    tridiag[(1, 0)] = beta
    vecs[1] = w / beta

    def reortho_step(j, state):
        vecs, w = state

        tau = vecs[j, :].reshape(shape_dtype_tuple[0])
        coeff = np.dot(w, tau)
        w -= coeff * tau
        return vecs, w

    def lanczos_step(i, state):
        tridiag, vecs, beta = state

        v = vecs[i, :].reshape(shape_dtype_tuple[0])
        v_old = vecs[i - 1, :].reshape(shape_dtype_tuple[0])

        w = mat(v) - beta * v_old
        alpha = np.dot(w, v)
        tridiag[(i, i)] = alpha
        w -= alpha * v

        # Full reorthogonalization
        for k in range(0, i):
            vecs, w = reortho_step(k, (vecs, w))

        # TODO: Raise if lanczos vectors are independent i.e. `beta` small?
        beta = np.linalg.norm(w)

        tridiag[(i, i + 1)] = beta
        tridiag[(i + 1, i)] = beta
        vecs[i + 1] = w / beta

        return tridiag, vecs, beta

    for k in range(1, order - 1):
        tridiag, vecs, beta = lanczos_step(k, (tridiag, vecs, beta))

    # Final tridiag value and reorthogonalization
    v = vecs[order - 1, :].reshape(shape_dtype_tuple[0])
    v_old = vecs[order - 2, :].reshape(shape_dtype_tuple[0])
    w = mat(v) - beta * v_old
    alpha = np.dot(w, v)
    tridiag[(order - 1, order - 1)] = alpha
    w -= alpha * v
    for k in range(0, order - 1):
        vecs, w = reortho_step(k, (vecs, w))

    return tridiag, vecs


def stochastic_logdet_from_lanczos(
    tridiag_stack: np.ndarray, matrix_shape0: int, func: Callable = np.log
):
    """Computes a stochastic estimate of the log-determinate of a matrix using
    its Lanczos decomposition.

    Implemented via the stoachstic Lanczos quadrature.
    """
    eig_vals, eig_vecs = np.linalg.eigh(tridiag_stack)
    # TODO: Mask Eigenvalues <= 0?

    num_random_probes = tridiag_stack.shape[0]

    print(eig_vecs[0].shape)
    print(eig_vecs[0])
    eig_ves_first_component = eig_vecs[..., 0, :]
    print(eig_ves_first_component[0])
    func_of_eig_vals = func(eig_vals)

    dot_products = np.sum(eig_ves_first_component**2 * func_of_eig_vals)
    return matrix_shape0 / float(num_random_probes) * dot_products


def stochastic_lq_logdet(
    mat: Union[np.ndarray, Callable],
    order: int,
    n_samples: int,
    rng: Generator,
    *,
    shape0: Optional[int] = None,
    dtype=None
):
    """Computes a stochastic estimate of the log-determinate of a matrix using
    the stoachstic Lanczos quadrature algorithm.
    """
    shape0 = shape0 if shape0 is not None else mat.shape[0]
    mat = mat.__matmul__ if not hasattr(mat, "__call__") else mat
    ss = rng.bit_generator._seed_seq
    keys = ss.spawn(n_samples)

    lanczos = partial(lanczos_tridiag, mat, ((shape0,), dtype))
    # tridiags, _ = jax.vmap(lanczos, in_axes=(None, 0),
    #                        out_axes=(0, 0))(order, keys)
    # tridiags = np.array(Parallel(n_jobs=2)(delayed(lanczos)(order, key) for key in keys))[:, 0]
    tridiags = []
    for key in keys:
        tridiags.append(lanczos(order, key))
    tridiags = np.array(tridiags)[:, 0]

    return stochastic_logdet_from_lanczos(tridiags, shape0)


if __name__ == "__main__":
    seed = 10
    shape0 = 5

    rng = np.random.default_rng(seed)
    seed_sequence = rng.bit_generator._seed_seq
    m = rng.normal(size=(shape0,) * 2)
    m = m @ m.T  # ensure positive-definiteness

    tridiag, vecs = lanczos_tridiag(partial(matmul, m), ((shape0,), np.float64), shape0, seed_sequence)
    m_est = vecs.T @ tridiag @ vecs

    np.testing.assert_allclose(m_est, m, atol=1e-13, rtol=1e-13)

    print(np.min(m))
    exit()
    print("logdet:", np.linalg.slogdet(m))

    print("Lanczos logdet:", stochastic_lq_logdet(m, shape0, 1000, rng))
