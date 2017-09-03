# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np

from ...spaces import RGSpace

from ..descriptors import Axis
from ..figures import Figure2D
from ..plots import Cartesian2D
from .plotter_base import PlotterBase


class RG1DPlotter(PlotterBase):
    def __init__(self, interactive=False, path='plot.html', line=None,
                 marker=None):
        super(RG1DPlotter, self).__init__(interactive, path)
        self.line = line
        self.marker = marker

    @property
    def domain_classes(self):
        return (RGSpace, )

    def _initialize_plot(self):
        return Cartesian2D(data=None)

    def _initialize_figure(self):
        xaxis = Axis()
        yaxis = Axis()
        return Figure2D(plots=None, xaxis=xaxis, yaxis=yaxis)

    def _parse_data(self, data, field, spaces):
        y_data = data
        rgspace = field.domain[spaces[0]]
        xy_data = np.empty((2, y_data.shape[0]))
        xy_data[1] = y_data
        num = rgspace.shape[0]
        length = rgspace.distances[0]*num
        xy_data[0] = np.linspace(start=0,
                                 stop=length,
                                 num=num,
                                 endpoint=False)

        return xy_data
