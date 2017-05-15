# -*- coding: utf-8 -*-
from abc import abstractmethod
from scatter_plot import ScatterPlot


class Cartesian(ScatterPlot):
    def __init__(self, x, y, label, line, marker, showlegend=True):
        super(Cartesian, self).__init__(label, line, marker)
        self.x = x
        self.y = y
        self.showlegend = showlegend

    @abstractmethod
    def to_plotly(self):
        plotly_object = super(Cartesian, self).to_plotly()
        plotly_object['x'] = self.x
        plotly_object['y'] = self.y
        plotly_object['showlegend'] = self.showlegend
        return plotly_object
