# -*- coding: utf-8 -*-

from nifty import dependency_injector as gdi
from heatmap import Heatmap
import numpy as np

from .mollweide_helper import mollweide_helper

pyHealpix = gdi.get('pyHealpix')


class HPMollweide(Heatmap):
    def __init__(self, data, xsize=800, color_map=None, webgl=False,
                 smoothing=False):  # smoothing 'best', 'fast', False
        if 'pyHealpix' not in gdi:
            raise ImportError(
                "The module pyHealpix is needed but not available.")
        self.xsize = xsize
        super(HPMollweide, self).__init__(data, color_map, webgl, smoothing)

    def at(self, data):
        if isinstance(data, list):
            data = [self._mollview(d) for d in data]
        else:
            data = self._mollview(data)
        return HPMollweide(data=data,
                           xsize=self.xsize,
                           color_map=self.color_map,
                           webgl=self.webgl,
                           smoothing=self.smoothing)

    def _mollview(self, x):
        xsize = self.xsize
        res, mask, theta, phi = mollweide_helper(xsize)

        ptg = np.empty((phi.size, 2), dtype=np.float64)
        ptg[:, 0] = theta
        ptg[:, 1] = phi
        base = pyHealpix.Healpix_Base(int(np.sqrt(x.size/12)), "RING")
        res[mask] = x[base.ang2pix(ptg)]
        return res