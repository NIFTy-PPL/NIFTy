from __future__ import division
from builtins import range
import numpy as np

from .endomorphic_operator import EndomorphicOperator
from .. import dobj
from ..spaces import PowerSpace
from .. import utilities
from .. import Field, DomainTuple


class DirectSmoothingOperator(EndomorphicOperator):
    def __init__(self, domain, sigma, log_distances=False,
                 space=None):
        super(DirectSmoothingOperator, self).__init__()

        self._domain = DomainTuple.make(domain)
        self._space = utilities.infer_space(self._domain, space)
        if not isinstance(self._domain[self._space], PowerSpace):
            raise TypeError("PowerSpace needed")

        self._sigma = float(sigma)
        self._log_distances = log_distances
        self._effective_smoothing_width = 3.01

        self._axis = self._domain.axes[self._space][0]
        if self._sigma != 0:
            distances = self._domain[self._space].k_lengths
            if self._log_distances:
                distances = np.log(np.maximum(distances, 1e-15))

            self._ibegin, self._nval, self._wgt = self._precompute(distances)

    def _times(self, x):
        if self._sigma == 0:
            return x.copy()

        return self._smooth(x)

    @property
    def domain(self):
        return self._domain

    @property
    def self_adjoint(self):
        return True

    @property
    def unitary(self):
        return False

    def _precompute(self, x):
        """ Does precomputations for Gaussian smoothing on a 1D irregular grid.

        Parameters
        ----------
        x: 1D floating point array or list containing the individual grid
            positions. Points must be given in ascending order.


        Returns
        -------
        ibegin: integer array of the same size as x
            ibegin[i] is the minimum grid index to consider when computing the
            smoothed value at grid index i
        nval: integer array of the same size as x
            nval[i] is the number of indices to consider when computing the
            smoothed value at grid index i.
        wgt: list with the same number of entries as x
            wgt[i] is an array with nval[i] entries containing the
            normalized smoothing weights.
        """
        dxmax = self._effective_smoothing_width*self._sigma

        x = np.asarray(x)

        ibegin = np.searchsorted(x, x-dxmax)
        nval = np.searchsorted(x, x+dxmax) - ibegin

        wgt = []
        expfac = 1. / (2. * self._sigma*self._sigma)
        for i in range(x.size):
            if nval[i] > 0:
                t = x[ibegin[i]:ibegin[i]+nval[i]]-x[i]
                t = np.exp(-t*t*expfac)
                t *= 1./np.sum(t)
                wgt.append(t)
            else:
                wgt.append(np.array([]))

        return ibegin, nval, wgt

    def _smooth(self, x):
        if dobj.distaxis(x.val) == self._axis:
            val = dobj.redistribute(x.val, nodist=(self._axis,))
        else:
            val = x.val.copy()
        lval = dobj.local_data(val)

        for sl in utilities.get_slice_list(lval.shape, (self._axis,)):
            inp = lval[sl]
            out = np.zeros(inp.shape[0], dtype=inp.dtype)
            for i in range(inp.shape[0]):
                out[self._ibegin[i]:self._ibegin[i]+self._nval[i]] += \
                    inp[i] * self._wgt[i][:]
            lval[sl] = out
        val = dobj.redistribute(val, dobj.distaxis(x.val))
        return Field(x.domain, val)
