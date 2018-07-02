import numpy as np

from ..domain_tuple import DomainTuple
from ..domains import PowerSpace, RGSpace
from ..field import Field
from .linear_operator import LinearOperator


class ExpTransform(LinearOperator):
    def __init__(self, target, dof):

        if not ((isinstance(target, RGSpace) and target.harmonic) or
                isinstance(target, PowerSpace)):
            raise ValueError(
                "Target must be a harmonic RGSpace or a power space.")

        if np.isscalar(dof):
            dof = np.full(len(target.shape), int(dof), dtype=np.int)
        dof = np.array(dof)
        ndim = len(target.shape)

        t_mins = np.empty(ndim)
        bindistances = np.empty(ndim)
        self._bindex = [None] * ndim
        self._frac = [None] * ndim

        for i in range(ndim):
            if isinstance(target, RGSpace):
                rng = np.arange(target.shape[i])
                tmp = np.minimum(rng, target.shape[i] + 1 - rng)
                k_array = tmp * target.distances[i]
            else:
                k_array = target.k_lengths

            # avoid taking log of first entry
            log_k_array = np.log(k_array[1:])

            # Interpolate log_k_array linearly
            t_max = np.max(log_k_array)
            t_min = np.min(log_k_array)

            # Save t_min for later
            t_mins[i] = t_min

            bindistances[i] = (t_max - t_min) / (dof[i] - 1)
            coord = np.append(0., 1. + (log_k_array - t_min) / bindistances[i])
            self._bindex[i] = np.floor(coord).astype(int)

            # Interpolated value is computed via
            # (1.-frac)*<value from this bin> + frac*<value from next bin>
            # 0 <= frac < 1.
            self._frac[i] = coord - self._bindex[i]

        from ..domains import LogRGSpace
        log_space = LogRGSpace(2 * dof + 1, bindistances,
                               t_mins, harmonic=False)
        self._target = DomainTuple.make(target)
        self._domain = DomainTuple.make(log_space)

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.to_global_data()
        ndim = len(self.target.shape)
        idx = ()
        for d in range(ndim):
            fst_dims = (1,) * d
            lst_dims = (1,) * (ndim - d - 1)
            wgt = self._frac[d].reshape(fst_dims + (-1,) + lst_dims)

            # ADJOINT_TIMES
            if mode == self.ADJOINT_TIMES:
                shp = list(x.shape)
                shp[d] = self._tgt(mode).shape[d]
                xnew = np.zeros(shp, dtype=x.dtype)
                np.add.at(xnew, idx + (self._bindex[d],), x * (1. - wgt))
                np.add.at(xnew, idx + (self._bindex[d] + 1,), x * wgt)

            # TIMES
            else:
                xnew = x[idx + (self._bindex[d],)] * (1. - wgt)
                xnew += x[idx + (self._bindex[d] + 1,)] * wgt

            x = xnew
            idx = (slice(None),) + idx
        return Field.from_global_data(self._tgt(mode), x)

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES
