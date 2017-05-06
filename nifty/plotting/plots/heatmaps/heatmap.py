# -*- coding: utf-8 -*-

from nifty.plotting.plotly_wrapper import PlotlyWrapper
import numpy as np

class Heatmap(PlotlyWrapper):
    def __init__(self, data, color_map=None, webgl=False,
                 smoothing=False):  # smoothing 'best', 'fast', False
        if isinstance(data, list):
            self.data = np.zeros((data[0].shape))
            for arr in data:
                self.data = np.add(self.data, arr)
        else:
            self.data = data
        self.color_map = color_map
        self.webgl = webgl
        self.smoothing = smoothing

    def to_plotly(self):
        plotly_object = dict()
        plotly_object['z'] = self.data
        plotly_object['showscale'] = False
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
