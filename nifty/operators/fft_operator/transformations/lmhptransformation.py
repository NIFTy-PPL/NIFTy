import numpy as np
from transformation import Transformation
from d2o import distributed_data_object
from nifty.config import dependency_injector as gdi
import nifty.nifty_utilities as utilities
from nifty import HPSpace, LMSpace

hp = gdi.get('healpy')


class LMHPTransformation(Transformation):
    def __init__(self, domain, codomain=None, module=None):
        if gdi.get('healpy') is None:
            raise ImportError(
                "The module libsharp is needed but not available.")

        if self.check_codomain(domain, codomain):
            self.domain = domain
            self.codomain = codomain
        else:
            raise ValueError("ERROR: Incompatible codomain!")

    @staticmethod
    def get_codomain(domain):
        """
            Generates a compatible codomain to which transformations are
            reasonable, i.e.\  a pixelization of the two-sphere.

            Parameters
            ----------
            domain : LMSpace
                Space for which a codomain is to be generated

            Returns
            -------
            codomain : HPSpace
                A compatible codomain.

            References
            ----------
            .. [#] K.M. Gorski et al., 2005, "HEALPix: A Framework for
                   High-Resolution Discretization and Fast Analysis of Data
                   Distributed on the Sphere", *ApJ* 622..759G.
        """
        if domain is None:
            raise ValueError('ERROR: cannot generate codomain for None')

        if not isinstance(domain, LMSpace):
            raise TypeError('ERROR: domain needs to be a LMSpace')

        nside = (domain.lmax + 1) // 3
        return HPSpace(nside=nside)

    @staticmethod
    def check_codomain(domain, codomain):
        if not isinstance(domain, LMSpace):
            raise TypeError('ERROR: domain is not a LMSpace')

        if codomain is None:
            return False

        if not isinstance(codomain, HPSpace):
            raise TypeError('ERROR: codomain must be a HPSpace.')
        nside = codomain.nside
        lmax = domain.lmax
        mmax = domain.mmax

        if (lmax != mmax) or (3 * nside - 1 != lmax):
            return False

        return True

    def transform(self, val, axes=None, **kwargs):
        """
        LM -> HP transform method.

        Parameters
        ----------
        val : np.ndarray or distributed_data_object
            The value array which is to be transformed

        axes : None or tuple
            The axes along which the transformation should take place

        """
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

            nside = self.codomain.nside
            lmax = self.domain.lmax
            mmax = self.domain.mmax

            inp = inp.astype(np.complex128, copy=False)
            inp = hp.alm2map(inp, nside, lmax=lmax, mmax=mmax,
                             pixwin=False, fwhm=0.0, sigma=None,
                             pol=True, inplace=False)

            if slice_list == [slice(None, None)]:
                return_val = inp
            else:
                return_val[slice_list] = inp

        # re-weight if discrete
        if self.codomain.discrete:
            val = self.codomain.weight(val, power=0.5, axes=axes)

        if isinstance(val, distributed_data_object):
            new_val = val.copy_empty(dtype=self.codomain.dtype)
            new_val.set_full_data(return_val, copy=False)
        else:
            return_val = return_val.astype(self.codomain.dtype, copy=False)

        return return_val
