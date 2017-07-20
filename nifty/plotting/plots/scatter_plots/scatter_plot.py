# -*- coding: utf-8 -*-

import abc
from nifty.plotting.plotly_wrapper import PlotlyWrapper
from nifty.plotting.descriptors import Marker,\
                                       Line


class ScatterPlot(PlotlyWrapper):
    def __init__(self, data, label, line, marker):
        self.data = data
        self.label = label
        self.line = line
        self.marker = marker
        if not self.line and not self.marker:
            self.marker = Marker()
            self.line = Line()

    @abc.abstractmethod
    def at(self, data):
        raise NotImplementedError

    @abc.abstractproperty
    def figure_dimension(self):
        raise NotImplementedError

    @abc.abstractmethod
    def to_plotly(self):
        ply_object = dict()
        ply_object['name'] = self.label
        if self.line and self.marker:
            ply_object['mode'] = 'lines+markers'
            ply_object['line'] = self.line.to_plotly()
            ply_object['marker'] = self.marker.to_plotly()
        elif self.line:
            ply_object['mode'] = 'line'
            ply_object['line'] = self.line.to_plotly()
        elif self.marker:
            ply_object['mode'] = 'markers'
            ply_object['marker'] = self.marker.to_plotly()

        return ply_object

    def default_width(self):
        return 700

    def default_height(self):
        return 700

    @abc.abstractmethod
    def default_axes(self):
        raise NotImplementedError
