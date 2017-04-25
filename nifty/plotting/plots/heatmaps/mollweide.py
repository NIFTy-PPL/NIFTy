# -*- coding: utf-8 -*-

import pylab

import healpy.projaxes as PA
import healpy.pixelfunc as pixelfunc

from heat_map import HeatMap


class Mollweide(HeatMap):
    def __init__(self, data, label='', line=None, marker=None, webgl=False,
                 smoothing=False):  # smoothing 'best', 'fast', False
        data = self._mollview(data)
        super(Mollweide, self).__init__(data, label, line, marker, webgl,
                                        smoothing)

    def _mollview(self, x, xsize=800):
        x = pixelfunc.ma_to_array(x)
        f = pylab.figure(None, figsize=(8.5, 5.4))
        extent = (0.02, 0.05, 0.96, 0.9)
        ax = PA.HpxMollweideAxes(f, extent)
        img = ax.projmap(x, nest=False, xsize=xsize)
        return img
