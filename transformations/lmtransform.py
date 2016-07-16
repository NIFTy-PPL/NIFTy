import numpy as np
from nifty import GLSpace, HPSpace
from nifty.config import about
import nifty.nifty_utilities as utilities
from transform import Transform
from d2o import distributed_data_object

class  LMTransform(Transform):
    """
        LMTransform for transforming to GL/HP space
    """

    def __init__(self, domain, codomain, module):
        self.domain = domain
        self.codomain = codomain
        self.module = module

    def _transform(self, val):
        if isinstance(self.codomain, GLSpace):
            # shorthand for transform parameters
            nlat = self.codomain.paradict['nlat']
            nlon = self.codomain.paradict['nlon']
            lmax = self.domain.paradict['lmax']
            mmax = self.paradict['mmax']

            if self.domain.dtype == np.dtype('complex64')
                val = self.module.alm2map_f(val, nlat=nlat, nlon=nlon,
                                            lmax=lmax, mmax=mmax, cl=False)
            else:
                val = self.module.alm2map(val, nlat=nlat, nlon=nlon,
                                          lmax=lmax, mmax=mmax, cl=False)
        elif isinstance(self.codomain, HPSpace):
            # shorthand for transform parameters
            nside = self.codomain.paradict['nside']
            lmax = self.domain.paradict['lmax']
            mmax = self.domain.paradict['mmax']

            val = val.astype(np.complex128, copy=False)
            val = self.module.alm2map(val, nside, lmax=lmax, mmax=mmax,
                                      pixwin=False, fwhm=0.0, sigma=None,
                                      pol=True, inplace=False)
        else:
            raise ValueError("ERROR: Unsupported transformation.")

        return val

    def transform(self, val, axes, **kwargs):
        if isinstance(val, distributed_data_object):
            temp_val = val.get_full_data()
        else:
            temp_val = val

        return_val = None

        for slice_list in utilities.get_slice_list(temp_val.shape, axes):
            if slice_list == [slice(None, None)]:
                inp = temp_val
            else:
                if return_val is None:
                    return_val = np.empty_like(temp_val)
                inp = temp_val[slice_list]

            inp = self._transform(inp)

            if slice_list == [slice(None, None)]:
                return_val = inp
            else:
                return_val[slice_list] = inp

        # re-weight if discrete
        if self.codomain.discrete:
            val = self.codomain.calc_weight(val, power=0.5)

        if isinstance(val, distributed_data_object):
            new_val = val.copy_empty(dtype=self.codomain.dtype)
            new_val.set_full_data(return_val, copy=False)
        else:
            return_val = return_val.astype(self.codomain.dtype, copy=False)

        return return_val