# -*- coding: utf-8 -*-

from .figure_from_plot import FigureFromPlot
from nifty.plotting.plots import Heatmap, HPMollweide, GLMollweide


class Figure2D(FigureFromPlot):
    def __init__(self, plots, title=None, width=None, height=None,
                 xaxis=None, yaxis=None):
        if plots is not None:
            width = width if width is not None else plots[0].default_width()
            height = \
                height if height is not None else plots[0].default_height()
            xaxis = xaxis if xaxis is not None else plots[0].default_axes()[0]
            yaxis = yaxis if yaxis is not None else plots[0].default_axes()[1]

            if isinstance(plots[0], Heatmap) and width is None and \
               height is None:
                (y, x) = plots[0].data.shape

                width = 500
                height = int(500*y/x)

                if isinstance(plots[0], GLMollweide) or \
                   isinstance(plots[0], HPMollweide):
                    xaxis = False if (xaxis is None) else xaxis
                    yaxis = False if (yaxis is None) else yaxis

        super(Figure2D, self).__init__(plots, title, width, height)
        self.xaxis = xaxis
        self.yaxis = yaxis

    def at(self, plots, title=None):
        title = title if title is not None else self.title
        return Figure2D(plots=plots,
                        title=title,
                        width=self.width,
                        height=self.height,
                        xaxis=self.xaxis,
                        yaxis=self.yaxis)

    def to_plotly(self):

        plotly_object = super(Figure2D, self).to_plotly()

        if self.xaxis or self.yaxis:
            plotly_object['layout']['scene']['aspectratio'] = {}
        if self.xaxis:
            plotly_object['layout']['xaxis'] = self.xaxis.to_plotly()
        elif not self.xaxis:
            plotly_object['layout']['xaxis'] = dict(
                                                autorange=True,
                                                showgrid=False,
                                                zeroline=False,
                                                showline=False,
                                                autotick=True,
                                                ticks='',
                                                showticklabels=False
                                            )
        if self.yaxis:
            plotly_object['layout']['yaxis'] = self.yaxis.to_plotly()
        elif not self.yaxis:
            plotly_object['layout']['yaxis'] = dict(showline=False)

        return plotly_object
