# -*- coding: utf-8 -*-

from cartesian import Cartesian


class Cartesian3D(Cartesian):
    def __init__(self, data, label='', line=None, marker=None,
                 showlegend=True):
        super(Cartesian3D, self).__init__(data, label, line, marker,
                                          showlegend)

    def at(self, data):
        return Cartesian3D(data=data,
                           label=self.label,
                           line=self.line,
                           marker=self.marker,
                           showlegend=self.showlegend)

    @property
    def figure_dimension(self):
        return 3

    def to_plotly(self):
        plotly_object = super(Cartesian3D, self).to_plotly()
        plotly_object['z'] = self.data[2]
        plotly_object['type'] = 'scatter3d'
        return plotly_object
