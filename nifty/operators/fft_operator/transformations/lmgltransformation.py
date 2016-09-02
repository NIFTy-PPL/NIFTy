import numpy as np
from transformation import Transformation
from d2o import distributed_data_object
from nifty.config import dependency_injector as gdi
import nifty.nifty_utilities as utilities
from nifty import GLSpace, LMSpace

import lm_transformation_factory as ltf
gl = gdi.get('libsharp_wrapper_gl')


class LMGLTransformation(Transformation):
    def __init__(self, domain, codomain=None, module=None):
        if gdi.get('libsharp_wrapper_gl') is None:
            raise ImportError(
                "The module libsharp is needed but not available.")

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
            .. [#] M. Reinecke and D. Sverre Seljebotn, 2013,
                   "Libsharp - spherical
                   harmonic transforms revisited";
                   `arXiv:1303.4945 <http://www.arxiv.org/abs/1303.4945>`_
        """
        if domain is None:
            raise ValueError('ERROR: cannot generate codomain for None')

        if not isinstance(domain, LMSpace):
            raise TypeError('ERROR: domain needs to be a LMSpace')

        if domain.dtype == np.dtype('complex64'):
            new_dtype = np.float32
        elif domain.dtype == np.dtype('complex128'):
            new_dtype = np.float64
        else:
            raise ValueError('ERROR: unsupported domain dtype')

        nlat = domain.lmax + 1
        nlon = domain.lmax * 2 + 1
        return GLSpace(nlat=nlat, nlon=nlon, dtype=new_dtype)

    @staticmethod
    def check_codomain(domain, codomain):
        if not isinstance(domain, LMSpace):
            raise TypeError('ERROR: domain is not a LMSpace')

        if codomain is None:
            return False

        if not isinstance(codomain, GLSpace):
            raise TypeError('ERROR: codomain must be a GLSpace.')

        nlat = codomain.nlat
        nlon = codomain.nlon
        lmax = domain.lmax
        mmax = domain.mmax

        if (lmax != mmax) or (nlat != lmax + 1) or (nlon != 2 * lmax + 1):
            return False

        return True

    def transform(self, val, axes=None, **kwargs):
        """
        LM -> GL transform method.

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

            nlat = self.codomain.nlat
            nlon = self.codomain.nlon
            lmax = self.domain.lmax
            mmax = self.mmax

            if inp.dtype >= np.dtype('complex64'):
                inpReal = np.real(inp)
                inpImag = np.imag(inp)
                inpReal = ltf.buildLm(inpReal,lmax=lmax)
                inpImag = ltf.buildLm(inpImag,lmax=lmax)
                inpReal = self.GlAlm2Map(inpReal, nlat=nlat, nlon=nlon,
                                 lmax=lmax, mmax=mmax, cl=False)
                inpImag = self.GlAlm2Map(inpImag, nlat=nlat, nlon=nlon,
                                 lmax=lmax, mmax=mmax, cl=False)
                inp = inpReal+inpImag*(1j)
            else:
                inp = ltf.buildLm(inp, lmax=lmax)
                inp = self.GlAlm2Map(inp, nlat=nlat, nlon=nlon,
                                   lmax=lmax, mmax=mmax, cl=False)

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

    def GlAlm2Map(self, inp, **kwargs):
        if inp.dtype == np.dtype('complex64'):
            return gl.alm2map_f(inp, kwargs)
        else:
            return gl.alm2map(inp, kwargs)
