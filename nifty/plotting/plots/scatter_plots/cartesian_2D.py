# -*- coding: utf-8 -*-

from cartesian import Cartesian


class Cartesian2D(Cartesian):
    def __init__(self, x=None, y=None, x_start=0, x_step=1,
                 label='', line=None, marker=None, showlegend=True, webgl=True):
        if y is None:
            raise Exception('Error: no y data to plot')
        if x is None:
            x = range(x_start, len(y) * x_step, x_step)
        super(Cartesian2D, self).__init__(x, y, label, line, marker, showlegend)
        self.webgl = webgl

    def to_plotly(self):
        plotly_object = super(Cartesian2D, self).to_plotly()
        if self.webgl:
            plotly_object['type'] = 'scattergl'
        else:
            plotly_object['type'] = 'scatter'

        return plotly_object
