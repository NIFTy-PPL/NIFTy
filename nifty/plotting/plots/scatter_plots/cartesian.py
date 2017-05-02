# -*- coding: utf-8 -*-
from abc import abstractmethod
from scatter_plot import ScatterPlot


class Cartesian(ScatterPlot):
    def __init__(self, x, y, label, line, marker):
        super(Cartesian, self).__init__(label, line, marker)
        self.x = x
        self.y = y

    @abstractmethod
    def to_plotly(self):
        plotly_object = super(Cartesian, self).to_plotly()
        plotly_object['x'] = self.x
        plotly_object['y'] = self.y
        return plotly_object
