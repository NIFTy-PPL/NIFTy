# -*- coding: utf-8 -*-

from nifty import dependency_injector as gdi
from heatmap import Heatmap

pylab = gdi.get('pylab')
healpy = gdi.get('healpy')


class Mollweide(Heatmap):
    def __init__(self, data, color_map=None, webgl=False,
                 smoothing=False):  # smoothing 'best', 'fast', False
        if 'pylab' not in gdi:
            raise ImportError("The module pylab is needed but not available.")
        if 'healpy' not in gdi:
            raise ImportError("The module healpy is needed but not available.")
        if isinstance(data, list):
            data = [self._mollview(d) for d in data]
        else:
            data = self._mollview(data)
        super(Mollweide, self).__init__(data, color_map, webgl, smoothing)

    def _mollview(self, x, xsize=800):
        x = healpy.pixelfunc.ma_to_array(x)
        f = pylab.figure(None, figsize=(8.5, 5.4))
        extent = (0.02, 0.05, 0.96, 0.9)
        ax = healpy.projaxes.HpxMollweideAxes(f, extent)
        img = ax.projmap(x, nest=False, xsize=xsize)
        return img
