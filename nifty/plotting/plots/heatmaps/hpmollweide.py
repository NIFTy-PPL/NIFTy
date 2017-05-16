# -*- coding: utf-8 -*-

from nifty import dependency_injector as gdi
from heatmap import Heatmap
import numpy as np

pylab = gdi.get('pylab')
pyHealpix = gdi.get('pyHealpix')


class HPMollweide(Heatmap):
    def __init__(self, data, color_map=None, webgl=False,
                 smoothing=False):  # smoothing 'best', 'fast', False
        if 'pylab' not in gdi:
            raise ImportError("The module pylab is needed but not available.")
        if 'pyHealpix' not in gdi:
            raise ImportError(
                "The module pyHealpix is needed but not available.")
        if isinstance(data, list):
            data = [self._mollview(d) for d in data]
        else:
            data = self._mollview(data)
        super(HPMollweide, self).__init__(data, color_map, webgl, smoothing)

    def _mollview(self, x, xsize=800):
        f = pylab.figure(None, figsize=(8.5, 5.4))
        extent = (0.02, 0.05, 0.96, 0.9)
        nside = int(np.sqrt(x.size//12))
        base = pyHealpix.Healpix_Base(nside, "RING")
        ra = np.linspace(-np.pi, np.pi, xsize)
        dec = np.linspace(-np.pi/2, np.pi/2, xsize/2)
        X, Y = np.meshgrid(ra, dec)
        dims = X.shape+(2,)
        ptg = np.empty(dims, dtype=np.float64)
        ptg[:, :, 0] = 0.5*np.pi-Y
        ptg[:, :, 1] = X+np.pi
        Z = x[base.ang2pix(ptg)]
        ax = f.add_subplot(111, projection='mollweide')
        ax.pcolormesh(X, Y, Z, rasterized=True)
        return f
