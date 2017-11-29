from .linear_operator import LinearOperator
from .. import Field
from ..spaces import DOFSpace


class DOFProjectionOperator(LinearOperator):
    def __init__(self, domain, dofdex, space=None):
        super(DOFProjectionOperator, self).__init__()

        self._domain = DomainTuple.make(domain)
        if space is None and len(self._domain) == 1:
            space = 0
        space = int(space)
        if space < 0 or space >= len(self.domain):
            raise ValueError("space index out of range")
        partner = self._domain[space]
        if not isinstance(dofdex, Field):
            raise TypeError("dofdex must be a Field")
        if not isinstance(dofdex.dtype, np.integer):
            raise TypeError("dofdex must contain integer numbers")
        if partner != dofdex.domain:
            raise ValueError("incorrect dofdex domain")

        nbin = dofdex.max()
        if partner.scalar_dvol() is not None:
            wgt = np.bincount(dobj.local_data(dofdex.val).ravel(),
                              minlength=nbin)
            wgt *= partner.scalar_dvol()
        else:
            dvol = dobj.local_data(partner.dvol())
            wgt = np.bincount(dobj.local_data(dofdex.val).ravel(),
                              minlength=nbin, weights=dvol)
        # The explicit conversion to float64 is necessary because bincount
        # sometimes returns its result as an integer array, even when
        # floating-point weights are present ...
        wgt = wgt.astype(np.float64, copy=False)
        wgt = dobj.np_allreduce_sum(wgt)
        if (wgt == 0).any():
            raise ValueError("empty bins detected")

        self._space = space
        tgt = list(self._domain)
        tgt[self._space] = DOFSpace(wgt)
        self._target = DomainTuple.make(tgt)

        if dobj.default_distaxis() in self.domain.axes[self._space]:
            dofdex = dobj.local_data(dofdex)
        else:  # dofdex must be available fully on every task
            dofdex = dobj.to_global_data(dofdex)
        self._dofdex = dofdex.ravel()
        firstaxis = self._domain.axes[self._space][0]
        lastaxis = self._domain.axes[self._space][-1]
        arrshape = dobj.local_shape(self._domain.shape, 0)
        presize = np.prod(arrshape[0:firstaxis], dtype=np.int)
        postsize = np.prod(arrshape[lastaxis+1:], dtype=np.int)
        self._hshape = (presize, self._target[self._space].shape[0], postsize)
        self._pshape = (presize, self._dofdex.size, postsize)

    def _times(self, x):
        arr = dobj.local_data(x.weight(1).val)
        arr = arr.reshape(self._pshape)
        oarr = np.zeros(self._hshape, dtype=x.dtype)
        np.add.at(oarr, (slice(None), self._dofdex, slice(None)), arr)
        if dobj.distaxis(x.val) in x.domain.axes[self._space]:
            oarr = dobj.np_allreduce_sum(oarr).reshape(self._target.shape)
            res = Field(self._target, dobj.from_global_data(oarr))
        else:
            oarr = oarr.reshape(dobj.local_shape(self._target.shape,
                                                 dobj.distaxis(x.val)))
            res = Field(self._target,
                        dobj.from_local_data(self._target.shape, oarr,
                                             dobj.default_distaxis()))
        return res.weight(-1, spaces=self._space)

    def _adjoint_times(self, x):
        res = Field.empty(self._domain, dtype=x.dtype)
        if dobj.distaxis(x.val) in x.domain.axes[self._space]:
            arr = dobj.to_global_data(x.val)
        else:
            arr = dobj.local_data(x.val)
        arr = arr.reshape(self._hshape)
        oarr = dobj.local_data(res.val).reshape(self._pshape)
        oarr[()] = arr[(slice(None), self._dofdex, slice(None))]
        return res

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def unitary(self):
        return False
