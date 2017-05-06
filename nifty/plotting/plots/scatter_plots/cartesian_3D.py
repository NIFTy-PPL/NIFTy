# -*- coding: utf-8 -*-

from cartesian import Cartesian


class Cartesian3D(Cartesian):
    def __init__(self, x, y, z, label='', line=None, marker=None, showlegend=True):
        super(Cartesian3D, self).__init__(x, y, label, line, marker, showlegend)
        self.z = z

    def to_plotly(self):
        plotly_object = super(Cartesian3D, self).to_plotly()
        plotly_object['z'] = self.z
        plotly_object['type'] = 'scatter3d'
        return plotly_object
