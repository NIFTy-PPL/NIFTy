from __future__ import division

import itertools
import numpy as np

from d2o import STRATEGIES as DISTRIBUTION_STRATEGIES

from nifty.spaces.space import Space
from nifty.config import about, nifty_configuration as gc,\
                         dependency_injector as gdi
from gl_space_paradict import GLSpaceParadict
import nifty.nifty_utilities as utilities

gl = gdi.get('libsharp_wrapper_gl')

GL_DISTRIBUTION_STRATEGIES = DISTRIBUTION_STRATEGIES['global']


class GLSpace(Space):
    """
        ..                 __
        ..               /  /
        ..     ____ __  /  /
        ..   /   _   / /  /
        ..  /  /_/  / /  /_
        ..  \___   /  \___/  space class
        .. /______/

        NIFTY subclass for Gauss-Legendre pixelizations [#]_ of the two-sphere.

        Parameters
        ----------
        nlat : int
            Number of latitudinal bins, or rings.
        nlon : int, *optional*
            Number of longitudinal bins (default: ``2*nlat - 1``).
        dtype : numpy.dtype, *optional*
            Data type of the field values (default: numpy.float64).

        See Also
        --------
        hp_space : A class for the HEALPix discretization of the sphere [#]_.
        lm_space : A class for spherical harmonic components.

        Notes
        -----
        Only real-valued fields on the two-sphere are supported, i.e.
        `dtype` has to be either numpy.float64 or numpy.float32.

        References
        ----------
        .. [#] M. Reinecke and D. Sverre Seljebotn, 2013, "Libsharp - spherical
               harmonic transforms revisited";
               `arXiv:1303.4945 <http://www.arxiv.org/abs/1303.4945>`_
        .. [#] K.M. Gorski et al., 2005, "HEALPix: A Framework for
               High-Resolution Discretization and Fast Analysis of Data
               Distributed on the Sphere", *ApJ* 622..759G.

        Attributes
        ----------
        para : numpy.ndarray
            One-dimensional array containing the two numbers `nlat` and `nlon`.
        dtype : numpy.dtype
            Data type of the field values.
        discrete : bool
            Whether or not the underlying space is discrete, always ``False``
            for spherical spaces.
        vol : numpy.ndarray
            An array containing the pixel sizes.
    """

    def __init__(self, nlat, nlon=None, dtype=np.dtype('float')):
        """
            Sets the attributes for a gl_space class instance.

            Parameters
            ----------
            nlat : int
                Number of latitudinal bins, or rings.
            nlon : int, *optional*
                Number of longitudinal bins (default: ``2*nlat - 1``).
            dtype : numpy.dtype, *optional*
                Data type of the field values (default: numpy.float64).

            Returns
            -------
            None

            Raises
            ------
            ImportError
                If the libsharp_wrapper_gl module is not available.
            ValueError
                If input `nlat` is invaild.

        """
        # check imports
        if not gc['use_libsharp']:
            raise ImportError(about._errors.cstring(
                "ERROR: libsharp_wrapper_gl not loaded."))

        # setup paradict
        self.paradict = GLSpaceParadict(nlat=nlat, nlon=nlon)

        # setup dtype
        self.dtype = np.dtype(dtype)

        # GLSpace is not harmonic
        self._harmonic = False

    @property
    def shape(self):
        return (np.int((self.paradict['nlat'] * self.paradict['nlon'])),)

    @property
    def dim(self):
        return np.int((self.paradict['nlat'] * self.paradict['nlon']))

    @property
    def total_volume(self):
        return 4 * np.pi

    def weight(self, x, power=1, axes=None, inplace=False):
        axes = utilities.cast_axis_to_tuple(axes, length=1)

        nlon = self.paradict['nlon']
        nlat = self.paradict['nlat']

        weight = np.array(list(itertools.chain.from_iterable(
                    itertools.repeat(x ** power, nlon)
                    for x in gl.vol(nlat))
                 ))

        if axes is not None:
            # reshape the weight array to match the input shape
            new_shape = np.ones(len(x.shape), dtype=np.int)
            for index in range(len(axes)):
                new_shape[index] = len(weight)
            weight = weight.reshape(new_shape)

        if inplace:
            x *= weight
            result_x = x
        else:
            result_x = x * weight

        return result_x
