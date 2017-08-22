# -*- coding: utf-8 -*-
from abc import abstractmethod
from .scatter_plot import ScatterPlot


class Cartesian(ScatterPlot):
    def __init__(self, data, label, line, marker, showlegend=True):
        super(Cartesian, self).__init__(data, label, line, marker)
        self.showlegend = showlegend

    @abstractmethod
    def to_plotly(self):
        plotly_object = super(Cartesian, self).to_plotly()
        plotly_object['x'] = self.data[0]
        plotly_object['y'] = self.data[1]
        plotly_object['showlegend'] = self.showlegend
        return plotly_object
