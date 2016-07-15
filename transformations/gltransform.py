import numpy as np
from transform import Transform
from d2o import distributed_data_object
from nifty.config import dependency_injector as gdi
import nifty.nifty_utilities as utilities


gl = gdi.get('libsharp_wrapper_gl')

class GLTransform(Transform):
    """
        GLTransform wrapper for libsharp's transform functions
    """

    def __init__(self, domain, codomain):
        self.domain = domain
        self.codomain = codomain

        if 'libsharp_wrapper_gl' not in gdi:
            raise ImportError("The module libsharp_wrapper_gl " +
                              "is needed but not available")

    def transform(self, val, axes, **kwargs):
        if self.domain.discrete:
            val = self.domain.calc_weight(val, power=-0.5)

        # shorthands for transform parameters
        nlat = self.domain.paradict['nlat']
        nlon = self.domain.paradict['nlon']
        lmax = self.codomain.paradict['lmax']
        mmax = self.codomain.paradict['mmax']

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

            if self.domain.dtype == np.dtype('float32'):
                inp = gl.map2alm_f(inp,
                                   nlat=nlat, nlon=nlon,
                                   lmax=lmax, mmax=mmax)
            else:
                inp = gl.map2alm(inp,
                                 nlat=nlat, nlon=nlon,
                                 lmax=lmax, mmax=mmax)

            if slice_list == [slice(None, None)]:
                return_val = inp
            else:
                return_val[slice_list] = inp

        if isinstance(val, distributed_data_object):
            new_val = val.copy_empty(dtype=self.codomain.dtype)
            new_val.set_full_data(return_val, copy=False)
        else:
            return_val = return_val.astype(self.codomain.dtype, copy=False)

        return return_val