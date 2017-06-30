# -*- coding: utf-8 -*-

from .cartesian import Cartesian


class Cartesian2D(Cartesian):
    def __init__(self, data, label='', line=None, marker=None, showlegend=True,
                 webgl=True):
        super(Cartesian2D, self).__init__(data, label, line, marker,
                                          showlegend)
        self.webgl = webgl

    def at(self, data):
        return Cartesian2D(data=data,
                           label=self.label,
                           line=self.line,
                           marker=self.marker,
                           showlegend=self.showlegend,
                           webgl=self.webgl)

    @property
    def figure_dimension(self):
        return 2

    def to_plotly(self):
        plotly_object = super(Cartesian2D, self).to_plotly()
        if self.webgl:
            plotly_object['type'] = 'scattergl'
        else:
            plotly_object['type'] = 'scatter'

        return plotly_object
