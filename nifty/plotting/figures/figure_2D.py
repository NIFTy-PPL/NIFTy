# -*- coding: utf-8 -*-

from figure_from_plot import FigureFromPlot
from nifty.plotting.plots import Heatmap, HPMollweide, GLMollweide


class Figure2D(FigureFromPlot):
    def __init__(self, plots, title=None, width=None, height=None,
                 xaxis=None, yaxis=None):
        """
        :param title: title of figure (subtitle when used as subfigure)
        :param xaxis: Nifty Axis, if folse, then not displayed (false default for Mollweide)
        :param yaxis: Nifty Axis, if folse, then not displayed (false default for Mollweide)
        """

        if plots is not None:
            if isinstance(plots[0], Heatmap) and width is None and \
                            height is None:
                (x, y) = plots[0].data.shape

                if x > y:
                    width = 500
                    height = int(500*y/x)
                else:
                    height = 500
                    width = int(500 * y / x)

                if isinstance(plots[0], GLMollweide) or \
                        isinstance(plots[0], HPMollweide):
                    xaxis = False if (xaxis is None) else xaxis
                    yaxis = False if (yaxis is None) else yaxis

            else:
                width = None
                height = None

        super(Figure2D, self).__init__(plots, title, width, height)
        self.xaxis = xaxis
        self.yaxis = yaxis

    def at(self, plots):
        return Figure2D(plots=plots,
                        title=self.title,
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
        elif self.xaxis is False:
            plotly_object['layout']['xaxis'] = dict(
                showgrid=False,
                zeroline=False,
                showline=False,
                ticks='',
                showticklabels=False
            )
        if self.yaxis:
            plotly_object['layout']['yaxis'] = self.yaxis.to_plotly()
        elif self.yaxis is False:
            plotly_object['layout']['yaxis'] = dict(
                showgrid=False,
                zeroline=False,
                showline=False,
                ticks='',
                showticklabels=False
            )

        return plotly_object
