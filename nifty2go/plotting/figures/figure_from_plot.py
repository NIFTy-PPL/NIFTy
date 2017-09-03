# -*- coding: utf-8 -*-

from .figure_base import FigureBase


class FigureFromPlot(FigureBase):
    def __init__(self, plots, title, width, height):
        super(FigureFromPlot, self).__init__(title, width, height)
        self.plots = plots

    def to_plotly(self):
        data = [plt.to_plotly() for plt in self.plots]
        layout = {'title': self.title,
                  'scene': {'aspectmode': 'cube'},
                  'autosize': False,
                  'width': self.width,
                  'height': self.height,
                  }
        plotly_object = {'data': data,
                         'layout': layout}

        return plotly_object
