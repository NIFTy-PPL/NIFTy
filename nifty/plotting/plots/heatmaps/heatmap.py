# -*- coding: utf-8 -*-

import numpy as np

from nifty.plotting.descriptors import Axis
from nifty.plotting.colormap import Colormap
from nifty.plotting.plotly_wrapper import PlotlyWrapper


class Heatmap(PlotlyWrapper):
    def __init__(self, data, color_map=None, webgl=False, smoothing=False,
                 zmin=None, zmax=None):
        # smoothing 'best', 'fast', False

        if color_map is not None:
            if not isinstance(color_map, Colormap):
                raise TypeError("Provided color_map must be an instance of "
                                "the NIFTy Colormap class.")
        self.color_map = color_map
        self.webgl = webgl
        self.smoothing = smoothing
        self.data = data
        self.zmin = zmin
        self.zmax = zmax
        self._font_size = 22
        self._font_family = 'Bento'

    def at(self, data):
        if isinstance(data, list):
            temp_data = np.zeros((data[0].shape))
            for arr in data:
                temp_data = np.add(temp_data, arr)
        else:
            temp_data = data
        return Heatmap(data=temp_data,
                       color_map=self.color_map,
                       webgl=self.webgl,
                       smoothing=self.smoothing,
                       zmin=self.zmin,
                       zmax=self.zmax)

    @property
    def figure_dimension(self):
        return 2

    def to_plotly(self):
        plotly_object = dict()

        plotly_object['z'] = self.data
        plotly_object['zmin'] = self.zmin
        plotly_object['zmax'] = self.zmax

        plotly_object['showscale'] = True
        plotly_object['colorbar'] = {'tickfont': {'size': self._font_size,
                                                  'family': self._font_family}}
        if self.color_map:
            plotly_object['colorscale'] = self.color_map.to_plotly()
        if self.webgl:
            plotly_object['type'] = 'heatmapgl'
        else:
            plotly_object['type'] = 'heatmap'
        if self.smoothing:
            plotly_object['zsmooth'] = self.smoothing
        return plotly_object

    def default_width(self):
        return 700

    def default_height(self):
        (x, y) = self.data.shape
        return int(700 * y / x)

    def default_axes(self):
        return (Axis(font_size=self._font_size),
                Axis(font_size=self._font_size))
