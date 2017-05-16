# -*- coding: utf-8 -*-

from nifty import dependency_injector as gdi
from heatmap import Heatmap
import numpy as np

pylab = gdi.get('pylab')
pyHealpix = gdi.get('pyHealpix')


class GLMollweide(Heatmap):
    def __init__(self, data, nlat, nlon, color_map=None, webgl=False,
                 smoothing=False):  # smoothing 'best', 'fast', False
        if 'pylab' not in gdi:
            raise ImportError("The module pylab is needed but not available.")
        if 'pyHealpix' not in gdi:
            raise ImportError(
                "The module pyHealpix is needed but not available.")
        if isinstance(data, list):
            data = [self._mollview(d) for d in data]
        else:
            data = self._mollview(data, nlat, nlon)
        super(GLMollweide, self).__init__(data, color_map, webgl, smoothing)

    def _find_closest(A, target):
        # A must be sorted
        idx = A.searchsorted(target)
        idx = np.clip(idx, 1, len(A)-1)
        left = A[idx-1]
        right = A[idx]
        idx -= target - left < right - target
        return idx

    def _mollview(self, x, nlat, nlon, xsize=800):
        f = pylab.figure(None, figsize=(8.5, 5.4))
        extent = (0.02, 0.05, 0.96, 0.9)
        x = np.reshape(x, (nlon, nlat))
        ra = np.linspace(-np.pi, np.pi, xsize)
        dec = np.linspace(-np.pi/2, np.pi/2, xsize/2)
        X, Y = np.meshgrid(ra, dec)
        gllat = pyHealpix.GL_thetas(nlat)-0.5*np.pi
        gllon = np.arange(nlon+1)*(2*np.pi/nlon)
        ilat = _find_closest(gllat, dec-0.5*np.pi)
        ilon = _find_closest(gllon, np.pi+ra)
        for i in range(ilon.size):
            if (ilon[i] == nlon):
                ilon[i] = 0
        Z = np.empty((xsize/2, xsize), dtype=np.float64)
        for i in range(xsize/2):
            Z[i, :] = x[ilon, ilat[i]]
        ax = f.add_subplot(111, projection='mollweide')
        ax.pcolormesh(X, Y, Z, rasterized=True)
        return f
