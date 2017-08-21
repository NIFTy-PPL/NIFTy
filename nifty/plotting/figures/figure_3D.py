# -*- coding: utf-8 -*-
from figure_from_plot import FigureFromPlot


class Figure3D(FigureFromPlot):
    def __init__(self, plots, title=None, width=None, height=None,
                 xaxis=None, yaxis=None, zaxis=None):
        if plots is not None:
            width = width if width is not None else plots[0].default_width()
            height = \
                height if height is not None else plots[0].default_height()
            xaxis = xaxis if xaxis is not None else plots[0].default_axes()[0]
            yaxis = yaxis if yaxis is not None else plots[0].default_axes()[1]
            zaxis = zaxis if zaxis is not None else plots[0].default_axes()[2]
        super(Figure3D, self).__init__(plots, title, width, height)
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.zaxis = zaxis

    def at(self, plots, title=None):
        title = title if title is not None else self.title
        return Figure3D(plots=plots,
                        title=title,
                        width=self.width,
                        height=self.height,
                        xaxis=self.xaxis,
                        yaxis=self.yaxis,
                        zaxis=self.zaxis)

    def to_plotly(self):
        plotly_object = super(Figure3D, self).to_plotly()
        if self.xaxis or self.yaxis or self.zaxis:
            plotly_object['layout']['scene']['aspectratio'] = dict()

        if self.xaxis:
            plotly_object['layout']['scene']['xaxis'] = self.xaxis.to_plotly()
        elif not self.xaxis:
            plotly_object['layout']['scene']['xaxis'] = dict(showline=False)

        if self.yaxis:
            plotly_object['layout']['scene']['yaxis'] = self.yaxis.to_plotly()
        elif not self.yaxis:
            plotly_object['layout']['scene']['yaxis'] = dict(showline=False)

        if self.zaxis:
            plotly_object['layout']['scene']['zaxis'] = self.zaxis.to_plotly()
        elif not self.zaxis:
            plotly_object['layout']['scene']['zaxis'] = dict(showline=False)

        return plotly_object
