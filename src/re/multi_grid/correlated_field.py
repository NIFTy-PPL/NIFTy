from dataclasses import field
from typing import Callable, List, Mapping, Union

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from scipy.special import j0, sici

from ..model import Model
from ..prior import LogNormalPrior, NormalPrior
from ..tree_math import ShapeWithDtype, zeros_like
from .indexing import Grid
from .kernel import ICRefine, _FrozenKernel
from .correlated_field_util import j1


def log_k_offset_dist(r_min, r_max, N):
    km = 1.0 / r_max
    kM = 1.0 / r_min
    dlk = (np.log(kM) - np.log(km)) / N
    return np.log(km), dlk


def logdists(r_min, r_max, N):
    return np.arange(N) * (np.log(r_max) - np.log(r_min)) / N + np.log(r_min)


def dists(r_min, r_max, N):
    ld = logdists(r_min, r_max, N)
    return np.concatenate([np.array([0.0]), np.exp(ld)])


def k_lengths(r_min, r_max, N):
    lkmin, dlk = log_k_offset_dist(r_min, r_max, N)
    lk = np.arange(N) * dlk + lkmin
    return np.concatenate((np.array([0.0]), np.exp(lk)))


def k_binbounds(r_min, r_max, N):
    lk = np.log(k_lengths(r_min, r_max, N)[1:])
    _, dlk = log_k_offset_dist(r_min, r_max, N)
    lk = np.append(lk - 0.5 * dlk, lk[-1] + 0.5 * dlk)
    return np.concatenate((np.array([0.0]), np.exp(lk)))


def norm_weights(r_min, r_max, N, d):
    k_bin = k_binbounds(r_min, r_max, N)
    if d == 1:
        fkr = sici(k_bin * r_max)[0]
    elif d == 2:
        fkr = 1.0 - j0(k_bin * r_max)
    elif d == 3:
        fkr = sici(k_bin * r_max)[0] - np.sin(k_bin * r_max)
    else:
        raise NotImplementedError
    res = fkr[1:] - fkr[:-1]
    if (d == 1) or (d == 3):
        res *= 2.0 / np.pi
    return res


def distfunc_from_specfunk(r_min, r_max, N, d, normalize=True):
    kl = k_lengths(r_min, r_max, N)
    f = distfunc_from_spec(r_min, r_max, N, d, normalize)

    def func(specfunc):
        spec = specfunc(kl)
        return f(spec)

    return func


def distfunc_from_spec(r_min, r_max, N, d, normalize=True):
    k_bin = k_binbounds(r_min, r_max, N)
    fct = [np.pi, 2.0 * np.pi, 2.0 * np.pi**2]
    if normalize:
        weights = norm_weights(r_min, r_max, N, d)

    def func(spec):
        if spec.size != N + 1:
            raise ValueError

        def distfunc(r):
            k = jnp.expand_dims(k_bin, tuple(i for i in range(len(r.shape))))
            r = r[..., jnp.newaxis]
            kr = r * k
            if d == 1:
                fkr = jnp.sin(kr)
            elif d == 2:
                fkr = kr * j1(kr)
            elif d == 3:
                fkr = jnp.sin(kr) - kr * jnp.cos(kr)
            else:
                raise NotImplementedError
            res0 = (k[..., 1:] ** d - k[..., :-1] ** d) / d
            resn0 = (fkr[..., 1:] - fkr[..., :-1]) / r**d
            res = jnp.where(r < 1e-10, res0, resn0) / fct[d - 1]
            res = jnp.tensordot(res, spec, axes=(-1, 0))
            if normalize:
                res /= (weights * spec).sum()
            return res

        return distfunc

    return func


def icrmatern_amplitude(
    rmin: float,
    rmax: float,
    Nbins: int,
    cutoff: Union[tuple, Callable],
    loglogslope: Union[tuple, Callable],
    prefix: str = "",
    kind: str = "amplitude",
) -> Model:
    """Constructs a function computing the amplitude of a Matérn-kernel
    power spectrum.

    See
    :class:`nifty8.re.correlated_field.CorrelatedFieldMaker.add_fluctuations
    _matern`
    for more details on the parameters.

    See also
    --------
    `Causal, Bayesian, & non-parametric modeling of the SARS-CoV-2 viral
    load vs. patient's age`, Guardiani, Matteo and Frank, Philipp and Kostić,
    Andrija and Edenhofer, Gordian and Roth, Jakob and Uhlmann, Berit and
    Enßlin, Torsten, `<https://arxiv.org/abs/2105.13483>`_
    `<https://doi.org/10.1371/journal.pone.0275011>`_
    """
    assert rmax > rmin
    assert rmin > 0
    if isinstance(cutoff, (tuple, list)):
        cutoff = LogNormalPrior(*cutoff, name=prefix + "cutoff")
    elif not callable(cutoff):
        te = f"invalid `cutoff` specified; got '{type(cutoff)}'"
        raise TypeError(te)
    if isinstance(loglogslope, (tuple, list)):
        loglogslope = NormalPrior(*loglogslope, name=prefix + "loglogslope")
    elif not callable(loglogslope):
        te = f"invalid `loglogslope` specified; got '{type(loglogslope)}'"
        raise TypeError(te)
    mode_lengths = k_lengths(rmin / rmax, 1.0, Nbins)

    def correlate(primals: Mapping) -> jnp.ndarray:
        ctf = cutoff(primals)
        slp = loglogslope(primals)

        ln_spectrum = 0.25 * slp * jnp.log1p((mode_lengths / ctf) ** 2)

        spectrum = jnp.exp(ln_spectrum)
        spectrum = spectrum.at[0].set(spectrum[1])

        if kind.lower() == "amplitude":
            spectrum = spectrum**2
        elif kind.lower() != "spectrum":
            raise ValueError(f"invalid kind specified {kind!r}")
        return spectrum

    model = Model(correlate, init=cutoff.init | loglogslope.init)
    model.klengths = mode_lengths
    return model


class ICRSpectral(Model):
    uindices: List[npt.NDArray] = field(metadata=dict(static=False))
    invindices: List[npt.NDArray] = field(metadata=dict(static=False))
    spectrum: Model = field(metadata=dict(static=False))

    def __init__(
        self,
        grid: Grid,
        spectrum: Model,
        rmin: float,
        rmax: float,
        dimension: int,
        offset,
        scale,
        window_size,
        rtol=1e-5,
        atol=1e-5,
        buffer_size=1000,
        use_distances=True,
        prefix="icr",
    ):
        assert rmax > rmin
        assert rmin > 0
        _get_kernelfunc = distfunc_from_spec(
            rmin / rmax, 1.0, spectrum.target.size - 1, dimension, normalize=False
        )

        def get_normalized_kerfunc(spec):
            func = _get_kernelfunc(spec)

            def kerfunc(x, y):
                r = jnp.linalg.norm(x - y, axis=0) / rmax
                return func(r) / func(jnp.zeros((1,)))[0]

            return kerfunc

        prefix = str(prefix)
        self._get_kernelfunc = get_normalized_kerfunc
        self.spectrum = spectrum
        self.grid = grid

        latent_kernel = self._get_kernelfunc(
            self.spectrum(zeros_like(self.spectrum.domain))
        )
        self.uindices, self.invindices, self.indexmaps = ICRefine(
            grid, latent_kernel, window_size
        )._freeze(
            rtol=rtol, atol=atol, buffer_size=buffer_size, use_distances=use_distances
        )

        self.window_size = window_size
        self.xikey = prefix + "xi"

        if isinstance(scale, (tuple, list)):
            scale = LogNormalPrior(*scale, name=prefix + "scale")
        elif not isinstance(scale, Model) or not isinstance(scale, float):
            raise ValueError
        self.scale = scale

        if isinstance(offset, (tuple, list)):
            offset = NormalPrior(*offset, name=prefix + "offset")
        elif not isinstance(offset, Model) or not isinstance(offset, float):
            raise ValueError
        self.offset = offset

        assert isinstance(spectrum.domain, dict)  # TODO use init
        grid_domain = {
            self.xikey + str(lvl): ShapeWithDtype(self.grid.at(lvl).shape, jnp.float_)
            for lvl in range(grid.depth + 1)
        }
        domain = spectrum.domain | grid_domain
        if isinstance(scale, Model):
            domain = domain | scale.domain
        if isinstance(offset, Model):
            domain = domain | offset.domain
        super().__init__(domain=domain, white_init=True)

    def get_kernel_function(self, x):
        spec = self.spectrum(x)
        return self._get_kernelfunc(spec)

    def get_kernel(self, x):
        kerfunc = self.get_kernel_function(x)
        kernel = ICRefine(self.grid, kerfunc, self.window_size)
        return _FrozenKernel(kernel, self.uindices, self.invindices, self.indexmaps)

    def __call__(self, x):
        kernel = self.get_kernel(x)
        scale = self.scale(x) if isinstance(self.scale, Model) else self.scale
        offset = self.offset(x) if isinstance(self.offset, Model) else self.offset
        xs = list(x[self.xikey + str(lvl)] for lvl in range(self.grid.depth + 1))
        res = kernel.apply(xs)[-1]
        res -= jnp.mean(res)
        res /= jnp.std(res)
        res *= scale
        res += offset
        return res

    def apply(self, x):
        return self(x)
