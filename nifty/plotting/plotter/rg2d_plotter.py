# -*- coding: utf-8 -*-


from nifty.spaces import RGSpace
from nifty.plotting.figures import Figure2D
from nifty.plotting.plots import Heatmap
from .plotter_base import PlotterBase


class RG2DPlotter(PlotterBase):
    def __init__(self, interactive=False, path='plot.html', color_map=None):
        self.color_map = color_map
        super(RG2DPlotter, self).__init__(interactive, path)

    @property
    def domain_classes(self):
        return (RGSpace, )

    def _initialize_plot(self):
        return Heatmap(data=None,
                       color_map=self.color_map)

    def _initialize_figure(self):
        return Figure2D(plots=None)

    def _parse_data(self, data, field, spaces):
        if len(data.shape) != 2:
            AttributeError("Only 2-dimensional RGSpaces are supported")
        return data
