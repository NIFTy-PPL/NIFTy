# -*- coding: utf-8 -*-

from abc import abstractmethod
from nifty.plotting.plotly_wrapper import PlotlyWrapper


class Plot(PlotlyWrapper):
    def __init__(self, label, line, marker):
        self.label = label
        self.line = line
        self.marker = marker

    @abstractmethod
    def to_plotly(self):
        ply_object = dict()
        ply_object['name'] = self.label
        if self.line and self.marker:
            ply_object['mode'] = 'lines+markers'
            ply_object['line'] = self.line.to_plotly()
            ply_object['marker'] = self.marker.to_plotly()
        elif self.line:
            ply_object['mode'] = 'markers'
            ply_object['line'] = self.line.to_plotly()
        elif self.marker:
            ply_object['mode'] = 'line'
            ply_object['marker'] = self.marker.to_plotly()

        return ply_object
