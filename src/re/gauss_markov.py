#!/usr/bin/env python3

# SPDX-License-Identifier: BSD-2-Clause
# Authors: Philipp Frank,

from functools import partial
from typing import Callable, Union

import jax.numpy as jnp
import numpy as np
from jax import Array, vmap
from jax.tree_util import tree_map
from jax.lax import fori_loop

from .model import Initializer, LazyModel, Model
from .prior import LogNormalPrior, NormalPrior
from .tree_math import ShapeWithDtype, random_like


def _isscalar(x):
    return jnp.ndim(x) == 0


def discrete_gauss_markov_process(
    xi: Array, x0: Array, drift: Array, diffamp: Array
):
    """Generator for a Gauss-Markov process  (GMP).

    Given the discrete transition probabilities via the `drift` and `diffamp`
    matrices, this function produces a series `res` of the form:

    .. math::
        res_{i+1} = drift_i @ res_i + diffamp_i @ xi_i

    which corresponds to a random realization of a GMP in case `xi` is
    standard normal distributed.

    Parameters:
    -----------
    xi: Array
        Input random variables for the GMP.
    x0: Array
        Initial state of the process. The first entry of the result `res_0` is
        always equal to `x0`.
    drift: Array
        Matrix or sequence of matrices that describe the mean of the transition
        probabilities of the GMP.
    diffamp: Array
        Matrix or sequence of matrices that correspond to the amplitude of the
        transition probabilities. For the GMP they imply that their matrix
        product `diffamp_i @ diffamp_i.T` is equal to the covariance of the
        transition probability iff `xi` is standard normal.

    Returns
    -------
        Sequence of vectors of the Gauss-Markov series. The sequence is returned
        as an array where the first axis corresponds to the steps of the
        sequence and the second axis are the entries of the vector.
    Notes:
    ------
    In case the sequence `xi` has length N (i.E. `xi.shape[0] = N`) the result
    sequence is of length N+1 as `x0` is returned as the first entry of the
    final sequence, followed by N transitions.
    """
    if _isscalar(drift):
        drift = drift * jnp.ones((1, 1))
    if _isscalar(diffamp):
        diffamp = diffamp * jnp.ones((1, 1))

    in_ax = (None if len(diffamp.shape) == 2 else 0, 0)
    res = vmap(jnp.matmul, in_ax, 0)(diffamp, xi)

    def loop(i, a):
        d = drift[i] if len(drift.shape) > 2 else drift
        return a.at[i + 1].add(jnp.matmul(d, a[i]))

    res = jnp.concatenate([x0[jnp.newaxis, ...], res], axis=0)
    return fori_loop(0, res.size, loop, res)


def scalar_gauss_markov_process(xi, x0, drift, diffamp):
    """Simple wrapper of `discrete_gm_general` for 1D scalar processes.
    """
    if not _isscalar(drift):
        drift = drift[:, jnp.newaxis, jnp.newaxis]
    if not _isscalar(diffamp):
        diffamp = diffamp[:, jnp.newaxis, jnp.newaxis]
    if _isscalar(x0):
        x0 = jnp.array([x0])
    return discrete_gauss_markov_process(
        xi[:, jnp.newaxis], x0, drift, diffamp
    )[:, 0]


def wiener_process(
    xi: Array,
    x0: Union[float, Array],
    sigma: Union[float, Array],
    dt: Union[float, Array],
):
    """Implements the Wiener process (WP)."""
    amp = jnp.sqrt(dt) * sigma
    return jnp.cumsum(jnp.concatenate((jnp.atleast_1d(x0).flatten(), amp * xi)))


def integrated_wiener_process(
    xi: Array,
    x0: Array,
    sigma: Array,
    dt: Array,
    asperity: Union[float, Array] = None
):
    """Implements the (generalized) Integrated Wiener process (IWP)."""
    asperity = 0. if asperity is None else asperity
    dt = jnp.ones(xi.shape[0]) * dt if _isscalar(dt) else dt
    res = (sigma * jnp.sqrt(dt))[:, jnp.newaxis] * xi
    res = res.at[:, 0].mul(jnp.sqrt(dt**2 / 12. + asperity))
    res = res.at[:, 0].add(0.5 * dt * res[:, 1])
    res = jnp.concatenate((x0[jnp.newaxis, ...], res), axis=0)
    res = res.at[:, 1].set(jnp.cumsum(res[:, 1]))
    res = res.at[1:, 0].add(dt * res[:-1, 1])
    return res.at[:, 0].set(jnp.cumsum(res[:, 0]))


def ornstein_uhlenbeck_process(
    xi: Array, x0: float, sigma: Union[float, Array],
    gamma: Union[float, Array], dt: Union[float, Array]
):
    """Implements the Ornstein Uhlenbeck process (OUP)."""
    drift = jnp.exp(-gamma * dt)
    amp = sigma * jnp.sqrt(1. - drift**2)
    return scalar_gauss_markov_process(xi, x0, drift, amp)


class GaussMarkovProcess(Model):
    def __init__(
        self,
        process: Callable,
        x0: Union[float, Array, LazyModel],
        dt: Union[float, Array],
        name='xi',
        N_steps: int = None,
        **kwargs
    ):
        if _isscalar(dt):
            if N_steps is None:
                msg = "`N_steps` is None and `dt` is not a sequence"
                raise NotImplementedError(msg)
            dt = np.ones(N_steps) * dt
        shp = dt.shape + jnp.shape(
            x0.target if isinstance(x0, LazyModel) else x0
        )
        domain = {name: ShapeWithDtype(shp)}
        init = Initializer(
            tree_map(lambda x: partial(random_like, primals=x), domain)
        )
        if isinstance(x0, LazyModel):
            domain = domain | x0.domain
            init = init | x0.init
        self.x0 = x0
        for _, a in kwargs.items():
            if isinstance(a, LazyModel):
                domain = domain | a.domain
                init = init | a.init
        self.kwargs = kwargs
        self.name = name
        self.process = process
        self.dt = dt

        super().__init__(domain=domain, init=init)

    def __call__(self, x):
        xi = x[self.name]
        xx = self.x0(x) if isinstance(self.x0, LazyModel) else self.x0
        tmp = {
            k: a(x) if isinstance(a, LazyModel) else a
            for k, a in self.kwargs.items()
        }
        return self.process(xi=xi, x0=xx, dt=self.dt, **tmp)


def WienerProcess(
    x0: Union[tuple, float, LazyModel],
    sigma: Union[tuple, float, Array, LazyModel],
    dt: Union[float, Array],
    name: str = 'wp',
    N_steps: int = None
):
    """Implements the Wiener process (WP).

    The WP in continuous time takes the form:

    .. math::
        d/dt x_t = sigma xi_t ,

    where `xi_t` is continuous time white noise.

    Parameters:
    -----------
    x0: tuple, float, or LazyModel
        Initial position of the WP. Can be passed as a fixed value, or a
        generative Model. Passing a tuple is a shortcut to set a normal prior
        with mean and std equal to the first and second entry of the tuple
        respectively on `x0`.
    sigma: tuple, float, Array, LazyModel
        Standard deviation of the WP. Analogously to `x0` may also be passed on
        as a model. May also be passed as a sequence of length equal to `dt` in
        which case a different sigma is used for each time interval.
    dt: float or Array of float
        Stepsizes of the process. In case it is a single float, `N_steps` must
        be provided to indicate the number of steps taken.
    name: str
        Name of the key corresponding to the parameters of the WP. Default `wp`.
    N_steps: int (optional)
        Option to set the number of steps in case `dt` is a scalar.

    Notes:
    ------
    In case `sigma` is time dependent, i.E. passed on as a sequence
    of length equal to `xi`, it is assumed to be constant within each timebin,
    i.E. `sigma_t = sigma_i for t_i <= t < t_{i+1}`.
    """
    if isinstance(x0, tuple):
        x0 = NormalPrior(x0[0], x0[1], name=name + '_x0')
    if isinstance(sigma, tuple):
        sigma = LogNormalPrior(sigma[0], sigma[1], name=name + '_sigma')
    return GaussMarkovProcess(
        wiener_process, x0, dt, name=name, N_steps=N_steps, sigma=sigma
    )


def IntegratedWienerProcess(
    x0: Union[tuple, Array, LazyModel],
    sigma: Union[tuple, float, Array, LazyModel],
    dt: Union[float, Array],
    name: str = 'iwp',
    asperity: Union[tuple, float, Array, LazyModel] = None,
    N_steps: int = None
):
    """Implements the Integrated Wiener Process.

    The generalized IWP in continuous time takes the form:

    .. math::
        d/dt x_t = y_t + sigma * asperity xi^1_t , \\
        d/dt y_t = sigma * xi^2_t

    where `xi^i_t` are continuous time white noise processes.
    This is a standard IWP in x in the case that `asperity` is zero (None). If
    it is non-zero, a WP with relative strength set by `asperity` is added to
    the dynamics of the IWP.

    For information on the parameters, please refer to `WienerProcess`. The
    `asperity` parameter may be defined analogously to `sigma`.

    This is also the process used in the CorrelatedField to describe the
    deviations of the power spectra from power-laws on a double logarithmic
    scale. See also `re.correlated_field.CorrelatedFieldMaker` and references
    there for further information.

    Notes:
    ------
    `sigma` and `asperity` may also be sequences. See notes on `WienerProcess`
    for further information.
    """
    if isinstance(x0, tuple):
        x0 = NormalPrior(x0[0], x0[1], shape=(2, ), name=name + '_x0')
    if isinstance(sigma, tuple):
        sigma = LogNormalPrior(sigma[0], sigma[1], name=name + '_sigma')
    if isinstance(asperity, tuple):
        asperity = LogNormalPrior(
            asperity[0], asperity[1], name=name + '_asperity'
        )
    return GaussMarkovProcess(
        integrated_wiener_process,
        x0,
        dt,
        name=name,
        N_steps=N_steps,
        sigma=sigma,
        asperity=asperity
    )


def OrnsteinUhlenbeckProcess(
    sigma: Union[tuple, float, Array, LazyModel],
    gamma: Union[tuple, float, Array, LazyModel],
    dt: Union[float, Array],
    name: str = 'oup',
    x0: Union[tuple, float, LazyModel] = None,
    N_steps: int = None
):
    """Implements the Ornstein Uhlenbeck process (OUP).

    The stochastic differential equation of the OUP takes the form:

    .. math::
        d/dt x_t + gamma x_t = sigma xi_t

    where `xi_t` is continuous time white noise.

    For information on the parameters, please refer to `WienerProcess`. The
    `gamma` parameter may be defined analogously to `sigma`.
    Unlike the WP and IWP, the OUP is a proper process, i.E. allows for a proper
    steady state solution. Therefore, `x0` may not be provided at all, in which
    case it defaults to a random variable drawn from the steady state
    distribution of the OUP.

    Notes:
    ------
    `sigma` and `gamma` may also be sequences. See notes on `WienerProcess`
    for further information.
    """
    if isinstance(sigma, tuple):
        sigma = LogNormalPrior(sigma[0], sigma[1], name=name + '_sigma')
    if isinstance(gamma, tuple):
        gamma = LogNormalPrior(gamma[0], gamma[1], name=name + '_gamma')
    if x0 is None:
        # x0 is not set, use steady state distribution of OUP to generate it.
        key = name + '_x0'

        def gen_x0(x):
            res = x[key]
            sig = sigma(x) if isinstance(sigma, LazyModel) else sigma
            return res * (sig if _isscalar(sig) else sig[0])

        domain = {key: ShapeWithDtype(())}
        init = Initializer(
            tree_map(lambda x: partial(random_like, primals=x), domain)
        )
        if isinstance(sigma, LazyModel):
            domain = domain | sigma.domain
            init = init | sigma.init
        x0 = Model(gen_x0, domain=domain, init=init)
    elif isinstance(x0, tuple):
        x0 = NormalPrior(x0[0], x0[1], name=name + '_x0')
    return GaussMarkovProcess(
        ornstein_uhlenbeck_process,
        x0,
        dt,
        name=name,
        N_steps=N_steps,
        sigma=sigma,
        gamma=gamma
    )
