# Copyright(C) 2023 Philipp Frank
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import jax.numpy as jnp
import numpy as np
from functools import partial, reduce
from scipy.special import sici, j0

from .convolve import charted_convolve, prepare_input
from .utils import j1
from .chart import MSChart
from .kernel import MSKernel
from ..model import AbstractModel, Model
from ..misc import ducktape
from ..tree_math import ShapeWithDtype


def _get_msc_shapewithdtype(chart, dtype):
    return list(ShapeWithDtype(ii.shape, dtype) for ii in chart.indices)


def MSConvolve(kernel):
    """Function to perform convolutions on an `MSChart` with an `MSKernel`.

    Parameters
    ----------
    kernel: MSKernel
        Kernel on `chart` that should be convolved with.
    percompute_kernel: bool (default True)
        Whether to precompute the kernel before application, or to evaluate the
        kernel function during application. If `False`, both the kernel
        evaluation and application to the input is lowered into a single `vmap`.
    Returns:
    --------
    function
        A python function that can be applied to an input of a list of
        jax.DeviceArray and applies the convultion with `kernel` on `chart`.
    """
    kernels = kernel.integral_kernels()
    return partial(charted_convolve,kernels=kernels, kerneltables=kernel.tables,
                   chart=kernel.chart)

def MSGp(kernel, dtype = jnp.float64):
    """Special case of `MSConvolve` that assumes the input to be standard
    normal random variables (see Notes) and `kernel` to be the amplitude of a
    Gaussian process.

    Parameters:
    -----------
    kernel: MSKernel
        Kernel that should be convolved with.
    percompute_kernel: bool (default True)
        Whether to precompute the kernel before application, or to evaluate the
        kernel function during application. If `False`, both the kernel
        evaluation and application to the input is lowered into a single `vmap`.
    dtype: jax.numpy.dtype (default float64)
        Dtype of the input to the convolution.
    Returns:
    --------
    nifty6.re.Model
        An instance of `Model` that can be applied to a `Vector` of standard 
        normal variables of consistent shape to obtain a random realization of 
        a GP with amplitude `kernel` on `chart`.
    Notes:
    ------
        In order to generate a consistent approximation of a continuous GP, the
        input random variables get scaled with the square root of the volume in
        each bin as this yields the correct variance of integrals over a 
        standard normal distributed random process.
    """
    f = MSConvolve(kernel)
    g = partial(prepare_input, chart=kernel.chart, volume_scaling = 0.5)
    return Model((lambda x: f(g(x))),
                 domain = _get_msc_shapewithdtype(kernel.chart, dtype))

def log_k_offset_dist(r_min, r_max, N):
    km = 1./r_max
    kM = 1./r_min
    dlk = (np.log(kM)-np.log(km))/N
    return np.log(km), dlk

def logdists(r_min, r_max, N):
    return np.arange(N)*(np.log(r_max)-np.log(r_min))/N + np.log(r_min)

def dists(r_min, r_max, N):
    ld = logdists(r_min, r_max, N)
    return np.concatenate([np.array([0.]), np.exp(ld)])

def k_lengths(r_min, r_max, N):
    lkmin, dlk = log_k_offset_dist(r_min, r_max, N)
    lk = np.arange(N)*dlk + lkmin
    return np.concatenate((np.array([0.]), np.exp(lk)))

def k_binbounds(r_min, r_max, N):
    lk = np.log(k_lengths(r_min, r_max, N)[1:])
    _, dlk = log_k_offset_dist(r_min, r_max, N)
    lk = np.append(lk-0.5*dlk, lk[-1]+0.5*dlk)
    return np.concatenate((np.array([0.]), np.exp(lk)))

def distfunc_from_spec(r_min, r_max, N, d, normalize = True):
    k_bin = k_binbounds(r_min, r_max, N)
    fct = [np.pi, 2.*np.pi, 2.*np.pi**2]
    if normalize:
        weights = norm_weights(r_min, r_max, N, d)

    def func(spec):
        if spec.size != N+1:
            raise ValueError
        def distfunc(r):
            k = jnp.expand_dims(k_bin, tuple(i for i in range(len(r.shape))))
            r = r[..., jnp.newaxis]
            kr = r*k
            if d == 1:
                fkr = jnp.sin(kr)
            elif d==2:
                fkr = kr*j1(kr)
            elif d==3:
                fkr = jnp.sin(kr) - kr*jnp.cos(kr)
            else:
                raise NotImplementedError
            res0 = (k[...,1:]**d - k[...,:-1]**d) / d
            resn0 = (fkr[...,1:] - fkr[..., :-1]) / r**d
            res = jnp.where(r < 1E-10, res0, resn0) / fct[d-1]
            res = jnp.tensordot(res, spec, axes=(-1,0)) 
            if normalize:
                res /= (weights*spec).sum()
            return res

        return distfunc
    return func

def norm_weights(r_min, r_max, N, d):
    k_bin = k_binbounds(r_min, r_max, N)
    if d == 1:
        fkr = sici(k_bin*r_max)[0]
    elif d == 2:
        fkr = 1.-j0(k_bin*r_max)
    elif d == 3:
        fkr = sici(k_bin*r_max)[0] - np.sin(k_bin*r_max)
    else:
        raise NotImplementedError
    res = (fkr[1:] - fkr[:-1])
    if (d == 1) or (d == 3):
        res *= (2./np.pi)
    return res

def distfunc_from_specfunk(r_min, r_max, N, d, normalize = True):
    kl = k_lengths(r_min, r_max, N)
    f = distfunc_from_spec(r_min, r_max, N, d, normalize)
    def func(specfunc):
        spec = specfunc(kl)
        return f(spec)
    return func

def distmat(*params):
    dsq = reduce(lambda a,b:a+b, (dd**2 for dd in params[len(params)//2:]))
    return jnp.sqrt(dsq[0])

def normal_transform(x, params):
    return params[0] + params[1]*x

def get_rminmax(chart):
    ker = MSKernel(distmat, chart)
    ds = ker.evaluate_kernel_function(0, chart.main_indices[0])
    r_max = np.max(ds)
    ds = ker.evaluate_kernel_function(chart.maxlevel, 
                                      chart.main_indices[chart.maxlevel])
    ds = ds.flatten()
    ds = ds[ds != 0.]
    r_min = np.min(ds)
    return r_min, r_max

class _MSSpectralGP(AbstractModel):
    def __init__(self, chart, specfunc, logamp, offset, offset_logamp, r_minmax, 
                 N, prefix = "", dtype = jnp.float64, stationary_axes = False,
                 scan_kernel = False, atol = 1E-5, rtol = 1E-5, 
                 buffer_size = 10000, nbatch = 10):
        """Abstract base class for isotropic GPs with a kernel defined via a 
        power spectrum.

        Parameters:
        -----------
        chart: MSChart
            Chart on which the GP is defined.
        specfunc: callable
            Function to obtain the spectrum in `N` bins with bounds defined via
            `r_minmax`. Takes an arbitrary pytree as an input and extracts the
            neccesary parameters via their keys. The
        logamp: float, tuple of float, or Model
            Parameters specifying the overall amplitude of the kernel. The
            kernel defined via `specfunc` gets normalized on the space defined
            by `chart` and `r_minmax` and re scaled with this overall amplitude.
            If `float` the amplitude is the exponential of `logamp`.
            If `tuple` the pair defines the mean and standard deviation of a
            Gaussian random variable that defines the log amplitude. This 
            parameter is inferred in addition to the other parameters and added
            to the domain of the GP.
            If `Model`
        #TODO
        dtype: jax.numpy.dtype (default float64)
            Dtype of the input to the convolution.
        Returns:
        --------
        nifty6.re.Model
            An instance of `Model` that can be applied to a `Vector` of standard 
            normal variables of consistent shape to obtain a random realization 
            of a GP with amplitude `kernel` on `chart`.
        Notes:
        ------
            In order to generate a consistent approximation of a continuous GP, 
            the input random variables get scaled with the square root of the 
            volume in each bin as this yields the correct variance of integrals
            over a standard normal distributed random process.
        """
        if not isinstance(chart, MSChart):
            raise ValueError
        self._chart = chart

        if isinstance(logamp, float):
            self._amp = lambda p: jnp.exp(logamp)
        else:
            key = prefix+'amplitude'
            if isinstance(logamp, tuple): 
                if not len(logamp) == 2:
                    raise ValueError
                self._pytree[key] = ShapeWithDtype(())
                self._amp = ducktape(lambda x: jnp.exp(
                    normal_transform(x, logamp)), key)
            elif isinstance(logamp, Model):
                if logamp.target != ShapeWithDtype(()):
                    raise ValueError
                self._pytree[key] = logamp.domain
                self._amp = ducktape(lambda x: jnp.exp(logamp(x)), key)
            else:
                raise ValueError
        offset = float(offset)
        if offset_logamp is None:
            self._off = lambda p: offset
        else:
            key = prefix+'offset'
            if isinstance(offset_logamp, float):
                self._pytree[key] = ShapeWithDtype(())
                self._off = lambda p: offset + jnp.exp(offset_logamp) * p[key]
            elif isinstance(offset_logamp, tuple):
                if not len(offset_logamp) == 2:
                    raise ValueError
                self._pytree[key] = ShapeWithDtype((2,), dtype)
                def off(x):
                    lamp = normal_transform(x[1], offset_logamp)
                    return offset + jnp.exp(lamp) * x[0]
                self._off = ducktape(off, key)
            elif isinstance(offset_logamp, Model):
                if offset_logamp.target != ShapeWithDtype(()):
                    raise ValueError
                self._pytree[key] = offset_logamp.domain
                self._off = ducktape(lambda x: offset_logamp(x), key)

        self._N = N
        self._r_minmax = r_minmax

        self._specfunc = specfunc
        self._ker_from_spec = distfunc_from_spec(self._r_minmax[0], 
                                                 self._r_minmax[1], N, 
                                                 self._chart.nspacedims, True)

        self._kernel = MSKernel(None, chart, stationary_axes, scan_kernel,
                                distmat, atol, rtol, buffer_size, nbatch)
        self._xikey = prefix+'xi'
        self._pytree[self._xikey] = _get_msc_shapewithdtype(chart, dtype)

    @property
    def kernel_dists(self):
        return dists(self._r_minmax[0], self._r_minmax[1], self._N)

    @property
    def k_lengths(self):
        return k_lengths(self._r_minmax[0], self._r_minmax[1], self._N)

    @property
    def domain(self):
        return self._pytree

    def get_amplitude(self, p):
        return self._amp(p)

    def get_offset(self, p):
        return self._off(p)

    def get_spec(self, p):
        return self._specfunc(p)

    def get_kernelfunc(self, p):
        spec = self._specfunc(p)
        normker = self._ker_from_spec(spec)
        amp = self.get_amplitude(p)

        return lambda *params: amp * normker(distmat(*params))

    def get_kernel(self, p):
        d = self.kernel_dists
        return self.get_kernelfunc(p)(d[np.newaxis, ...])

    def apply(self, p):
        kerfunc = self.get_kernelfunc(p)
        self._kernel.update_kernelfunction(kerfunc)
        off = self.get_offset(p)
        return list(r + off for r in MSGp(self._kernel)(p[self._xikey]))

class MSCorrelatedField(_MSSpectralGP):
    def __init__(self, chart, logamp, slope, logflex, offset_logamp, 
                 offset = 0., prefix = '', dtype = jnp.float64, r_minmax=None,
                 N = 50, stationary_axes = False, scan_kernel = False,
                 atol = 1E-5, rtol = 1E-5, buffer_size = 10000, nbatch = 10):
        """Special case of `MSConvolve` that assumes the input to be standard
        normal random variables (see Notes) and `kernel` to be the amplitude of a
        Gaussian process.

        Parameters:
        -----------
        kernel: MSKernel
            Kernel on `chart` that should be convolved with.
        dtype: jax.numpy.dtype (default float64)
            Dtype of the input to the convolution.
        Returns:
        --------
        nifty6.re.Model
            An instance of `Model` that can be applied to a `Vector` of standard 
            normal variables of consistent shape to obtain a random realization of 
            a GP with amplitude `kernel` on `chart`.
        Notes:
        ------
            In order to generate a consistent approximation of a continuous GP, the
            input random variables get scaled with the square root of the volume in
            each bin as this yields the correct variance of integrals over a 
            standard normal distributed random process.
        """
        self._pytree = {}
        if not isinstance(chart, MSChart):
            raise ValueError
        if r_minmax is None:
            self._r_minmax = get_rminmax(chart)
        else:
            self._r_minmax = r_minmax
        
        if isinstance(slope, float):
            self._slope = lambda p: slope
        else:
            key = prefix+'slope'
            if isinstance(slope, tuple):
                if not len(slope) == 2:
                    raise ValueError
                self._pytree[key] = ShapeWithDtype(())
                self._slope = ducktape(lambda x: normal_transform(x, slope), 
                                       key)
            elif isinstance(slope, Model):
                if slope.target != ShapeWithDtype(()):
                    raise ValueError
                self._pytree[key] = slope.domain
                self._slope = ducktape(slope, key)
            else:
                raise ValueError

        lks = k_lengths(self._r_minmax[0], self._r_minmax[1], N)
        lks = np.log(lks[1:])
        lm, dk = log_k_offset_dist(self._r_minmax[0], self._r_minmax[1], N)
        lks = np.concatenate([np.array([lm-dk]), lks])
        sigk = np.sqrt(dk)

        if logflex is None:
            get_dev = lambda p: 0.
        else:
            spkey = prefix+'spectralxi'
            self._pytree[spkey] = ShapeWithDtype((N,), dtype)
            if isinstance(logflex, float):
                self._flex = lambda p: jnp.exp(logflex)
            else:
                key = prefix+'flexibility'
                if isinstance(logflex, tuple):
                    if not len(logflex) == 2:
                        raise ValueError
                    self._flex = ducktape(lambda x: 
                                          jnp.exp(normal_transform(x, logflex)),
                                          key)
                    self._pytree[key] = ShapeWithDtype(())
                elif isinstance(logflex, Model):
                    if logflex.target != ShapeWithDtype(()):
                        raise ValueError
                    self._pytree[key] = logflex.domain
                    self._flex = ducktape(lambda x: jnp.exp(logflex(x)), key)

            def get_dev(p):
                flex = self.get_flex(p)
                xispec = p[spkey]
                return jnp.concatenate([jnp.array([0.,]), 
                                        flex*sigk*jnp.cumsum(xispec)])

        def get_spec(p):
            sl = self.get_slope(p)
            spec = sl * lks + get_dev(p)
            return jnp.exp(0.5 * spec)

        super().__init__(chart, get_spec, logamp, offset, offset_logamp,
                         self._r_minmax, N, prefix, dtype, stationary_axes, 
                         scan_kernel, atol, rtol, buffer_size, nbatch)

    def get_slope(self, p):
        return self._slope(p)

    def get_flex(self, p):
        return self._flex(p)

class MSMatern(_MSSpectralGP):
    def __init__(self, chart, logamp, slope, logscale, offset_logamp, 
                 offset = 0., prefix = '', dtype = jnp.float64, r_minmax=None, 
                 N = 50, stationary_axes = False, scan_kernel = False,
                 atol = 1E-5, rtol = 1E-5, buffer_size = 10000, nbatch = 10):
        self._pytree = {}
        if not isinstance(chart, MSChart):
            raise ValueError
        self._chart = chart
        if r_minmax is None:
            self._r_minmax = get_rminmax(self._chart)
        else:
            self._r_minmax = r_minmax

        if isinstance(slope, float):
            self._slope = lambda p: slope
        else:
            key = prefix+'slope'
            if isinstance(slope, tuple):
                if not len(slope) == 2:
                    raise ValueError
                self._pytree[key] = ShapeWithDtype(())
                self._slope = ducktape(lambda x: normal_transform(x, slope), 
                                       key)
            elif isinstance(slope, Model):
                if slope.target != ShapeWithDtype(()):
                    raise ValueError
                self._pytree[key] = slope.domain
                self._slope = ducktape(slope, key)
            else:
                raise ValueError

        if isinstance(logscale, float):
            self._scale = lambda p: jnp.exp(logscale)
        else:
            key = prefix+'logscale'
            if isinstance(logscale, tuple):
                if not len(logscale) == 2:
                    raise ValueError
                self._pytree[key] = ShapeWithDtype(())
                self._scale= ducktape(lambda x:
                                      jnp.exp(normal_transform(x, logscale)), 
                                      key)
            elif isinstance(logscale, Model):
                if logscale.target != ShapeWithDtype(()):
                    raise ValueError
                self._pytree[key] = logscale.domain
                self._scale = ducktape(lambda x: jnp.exp(logscale(x)), key)
            else:
                raise ValueError

        ks = k_lengths(self._r_minmax[0], self._r_minmax[1], N)
        def get_spec(p):
            sl = self.get_slope(p)
            scale = self.get_scale(p)
            res = 1. + (ks / scale)**2
            return res**(sl / 4.)

        super().__init__(chart, get_spec, logamp, offset, offset_logamp,
                         self._r_minmax, N, prefix, dtype, stationary_axes, 
                         scan_kernel, atol, rtol, buffer_size, nbatch = 10)

    def get_slope(self, p):
        return self._slope(p)

    def get_scale(self, p):
        return self._scale(p)