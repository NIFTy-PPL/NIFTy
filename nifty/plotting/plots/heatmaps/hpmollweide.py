# -*- coding: utf-8 -*-

from nifty import dependency_injector as gdi
from heatmap import Heatmap
import numpy as np

pyHealpix = gdi.get('pyHealpix')


class HPMollweide(Heatmap):
    def __init__(self, data, color_map=None, webgl=False,
                 smoothing=False):  # smoothing 'best', 'fast', False
        if 'pyHealpix' not in gdi:
            raise ImportError(
                "The module pyHealpix is needed but not available.")
        if isinstance(data, list):
            data = [self._mollview(d) for d in data]
        else:
            data = self._mollview(data)
        super(HPMollweide, self).__init__(data, color_map, webgl, smoothing)

    def _mollview(self, x, xsize=800):
        xsize = int(xsize)
        ysize = int(xsize/2)
        res = np.full(shape=(ysize,xsize), fill_value=np.nan, dtype=np.float64)
        xc = (xsize-1)*0.5
        yc = (ysize-1)*0.5
        i, j = np.meshgrid(np.arange(xsize), np.arange(ysize))
        u = 2*(i-xc)/(xc/1.02)
        v = (j-yc)/(yc/1.02)

        mask = np.where((u*u*0.25 + v*v) <= 1.)
        t1 = v[mask]
        theta = 0.5*np.pi-(
            np.arcsin(2/np.pi*(np.arcsin(t1) + t1*np.sqrt((1.-t1)*(1+t1)))))
        phi = -0.5*np.pi*u[mask]/np.maximum(np.sqrt((1-t1)*(1+t1)), 1e-6)
        ptg = np.empty((phi.size, 2), dtype=np.float64)
        ptg[:, 0] = theta
        ptg[:, 1] = phi
        base = pyHealpix.Healpix_Base(int(np.sqrt(x.size/12)), "RING")
        res[mask]=x[base.ang2pix(ptg)]
        return res
