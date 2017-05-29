# -*- coding: utf-8 -*-
import numpy as np

from nifty.plotting.colormap import Colormap
from nifty.plotting.plotly_wrapper import PlotlyWrapper


class Heatmap(PlotlyWrapper):
    def __init__(self, data, color_map=None, webgl=False, smoothing=False):
        """
        :param data: 2D array to plot
        :param color_map: Nifty Colormap, None as default
        :param webgl: optimized for webgl
        :param smoothing: 'best', 'fast', False
        """

        if color_map is not None:
            if not isinstance(color_map, Colormap):
                raise TypeError("Provided color_map must be an instance of "
                                "the NIFTy Colormap class.")
        self.color_map = color_map
        self.webgl = webgl
        self.smoothing = smoothing
        self.data = data

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
                       smoothing=self.smoothing)

    @property
    def figure_dimension(self):
        return 2

    def to_plotly(self):
        plotly_object = dict()

        plotly_object['z'] = self.data

        if self.color_map:
            plotly_object['colorscale'] = self.color_map.to_plotly()
            plotly_object['colorbar'] = dict(title=self.color_map.name, x=0.42)
        if self.webgl:
            plotly_object['type'] = 'heatmapgl'
        else:
            plotly_object['type'] = 'heatmap'
        if self.smoothing:
            plotly_object['zsmooth'] = self.smoothing
        return plotly_object
