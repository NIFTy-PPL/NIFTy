# -*- coding: utf-8 -*-

from nifty.plotting.plots.plot import Plot


class Heatmap(Plot):
    def __init__(self, data, label='', line=None, marker=None, webgl=False,
                 smoothing=False):  # smoothing 'best', 'fast', False
        super(Heatmap, self).__init__(label, line, marker)
        self.data = data
        self.webgl = webgl
        self.smoothing = smoothing

    def to_plotly(self):
        plotly_object = super(Heatmap, self).to_plotly()
        plotly_object['z'] = self.data
        if self.webgl:
            plotly_object['type'] = 'heatmapgl'
        else:
            plotly_object['type'] = 'heatmap'
        if self.smoothing:
            plotly_object['zsmooth'] = self.smoothing
        return plotly_object
