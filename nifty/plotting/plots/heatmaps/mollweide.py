# -*- coding: utf-8 -*-

from nifty import dependency_injector as gdi
from heat_map import HeatMap

pylab = gdi.get('pylab')
healpy = gdi.get('healpy')


class Mollweide(HeatMap):
    def __init__(self, data, label='', line=None, marker=None, webgl=False,
                 smoothing=False):  # smoothing 'best', 'fast', False
        if 'pylab' not in gdi:
            raise ImportError("The module pylab is needed but not available.")
        if 'healpy' not in gdi:
            raise ImportError("The module healpy is needed but not available.")

        data = self._mollview(data)
        super(Mollweide, self).__init__(data, label, line, marker, webgl,
                                        smoothing)

    def _mollview(self, x, xsize=800):
        x = healpy.pixelfunc.ma_to_array(x)
        f = pylab.figure(None, figsize=(8.5, 5.4))
        extent = (0.02, 0.05, 0.96, 0.9)
        ax = healpy.projaxes.HpxMollweideAxes(f, extent)
        img = ax.projmap(x, nest=False, xsize=xsize)
        return img
