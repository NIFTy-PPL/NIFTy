# -*- coding: utf-8 -*-

import numpy as np

from nifty.spaces import PowerSpace

from nifty.plotting.descriptors import Axis
from nifty.plotting.figures import Figure2D
from nifty.plotting.plots import Cartesian2D
from .plotter_base import PlotterBase


class PowerPlotter(PlotterBase):
    def __init__(self, interactive=False, path='plot.html', line=None,
                 marker=None):
        super(PowerPlotter, self).__init__(interactive, path)
        self.line = line
        self.marker = marker

    @property
    def domain_classes(self):
        return (PowerSpace, )

    def _initialize_plot(self):
        return Cartesian2D(data=None)

    def _initialize_figure(self):
        xaxis = Axis(log=True)
        yaxis = Axis(log=True)
        return Figure2D(plots=None, xaxis=xaxis, yaxis=yaxis)

    def _parse_data(self, data, field, spaces):
        y_data = data
        power_space = field.domain[spaces[0]]
        xy_data = np.empty((2, y_data.shape[0]))
        xy_data[1] = y_data
        xy_data[0] = power_space.kindex
        return xy_data
