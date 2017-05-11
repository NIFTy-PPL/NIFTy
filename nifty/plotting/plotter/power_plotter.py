# -*- coding: utf-8 -*-

from nifty.spaces import PowerSpace

from nifty.plotting.descriptors import Axis
from nifty.plotting.figures import Figure2D
from nifty.plotting.plots import Cartesian2D
from .plotter import Plotter


class PowerPlotter(Plotter):
    def __init__(self, interactive=False, path='.', title="",
                 line=None, marker=None):
        super(PowerPlotter, self).__init__(interactive, path, title)
        self.line = line
        self.marker = marker

    @property
    def domain_classes(self):
        return (PowerSpace, )

    def _create_individual_figure(self, plots):
        xaxis = Axis(log=True)
        yaxis = Axis(log=True)
        return Figure2D(plots, xaxis=xaxis, yaxis=yaxis)

    def _create_individual_plot(self, data, plot_domain):
        x = plot_domain[0].kindex
        result_plot = Cartesian2D(x=x,
                                  y=data)
        return result_plot
