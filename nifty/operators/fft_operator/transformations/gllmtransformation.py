import numpy as np
from transformation import Transformation
from d2o import distributed_data_object
from nifty.config import dependency_injector as gdi
import nifty.nifty_utilities as utilities
from nifty import GLSpace, LMSpace
import lm_transformation_factory as ltf

gl = gdi.get('libsharp_wrapper_gl')


class GLLMTransformation(Transformation):
    def __init__(self, domain, codomain=None, module=None):
        if 'libsharp_wrapper_gl' not in gdi:
            raise ImportError("The module libsharp is needed but not available")

        if codomain is None:
            self.domain = domain
            self.codomain = self.get_codomain(domain)
        elif self.check_codomain(domain, codomain):
            self.domain = domain
            self.codomain = codomain
        else:
            raise ValueError("ERROR: Incompatible codomain!")

    @staticmethod
    def get_codomain(domain):
        """
            Generates a compatible codomain to which transformations are
            reasonable, i.e.\  an instance of the :py:class:`lm_space` class.

            Parameters
            ----------
            domain: GLSpace
                Space for which a codomain is to be generated

            Returns
            -------
            codomain : LMSpace
                A compatible codomain.
        """
        if domain is None:
            raise ValueError('ERROR: cannot generate codomain for None')

        if not isinstance(domain, GLSpace):
            raise TypeError('ERROR: domain needs to be a GLSpace')

        nlat = domain.nlat
        lmax = nlat - 1
        mmax = nlat - 1
        if domain.dtype == np.dtype('float32'):
            return LMSpace(lmax=lmax, mmax=mmax, dtype=np.complex64)
        else:
            return LMSpace(lmax=lmax, mmax=mmax, dtype=np.complex128)

    @staticmethod
    def check_codomain(domain, codomain):
        if not isinstance(domain, GLSpace):
            raise TypeError('ERROR: domain is not a GLSpace')

        if codomain is None:
            return False

        if not isinstance(codomain, LMSpace):
            raise TypeError('ERROR: codomain must be a LMSpace.')

        nlat = domain.nlat
        nlon = domain.nlon
        lmax = codomain.lmax
        mmax = codomain.mmax

        if (nlon != 2 * nlat - 1) or (lmax != nlat - 1) or (lmax != mmax):
            return False

        return True

    def transform(self, val, axes=None, **kwargs):
        """
        GL -> LM transform method.

        Parameters
        ----------
        val : np.ndarray or distributed_data_object
            The value array which is to be transformed

        axes : None or tuple
            The axes along which the transformation should take place

        """
        if self.domain.discrete:
            val = self.domain.weight(val, power=-0.5, axes=axes)

        # shorthands for transform parameters
        nlat = self.domain.nlat
        nlon = self.domain.nlon
        lmax = self.codomain.lmax
        mmax = self.codomain.mmax

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

            if inp.dtype >= np.dtype('complex64'):
                inpReal = self.GlMap2Alm(
                    np.real(inp).astype(np.float64, copy=False), nlat=nlat,
                    nlon=nlon, lmax=lmax, mmax=mmax)
                inpImg = self.GlMap2Alm(
                    np.imag(inp).astype(np.float64, copy=False), nlat=nlat,
                    nlon=nlon, lmax=lmax, mmax=mmax)
                inpReal = ltf.buildIdx(inpReal, lmax=lmax)
                inpImg = ltf.buildIdx(inpImg, lmax=lmax)
                inp = inpReal + inpImg * 1j
            else:
                inp = self.GlMap2Alm(inp,
                                     nlat=nlat, nlon=nlon,
                                     lmax=lmax, mmax=mmax)
                inp = ltf.buildIdx(inp, lmax=lmax)

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

    def GlMap2Alm(self, inp, **kwargs):
        if inp.dtype == np.dtype('float32'):
            return gl.map2alm_f(inp, kwargs)
        else:
            return gl.map.alm(inp, kwargs)
