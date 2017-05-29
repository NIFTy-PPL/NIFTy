# -*- coding: utf-8 -*-
from figure_from_plot import FigureFromPlot


class Figure3D(FigureFromPlot):
    def __init__(self, plots, title=None, width=None, height=None,
                 xaxis=None, yaxis=None, zaxis=None):
        super(Figure3D, self).__init__(plots, title, width, height)
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.zaxis = zaxis

    def at(self, plots):
        return Figure3D(plots=plots,
                        title=self.title,
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
            plotly_object['layout']['scene']['xaxis'] = dict(
                showgrid=False,
                zeroline=False,
                showline=False,
                ticks='',
                showticklabels=False
            )

        if self.yaxis:
            plotly_object['layout']['scene']['yaxis'] = self.yaxis.to_plotly()
        elif not self.yaxis:
            plotly_object['layout']['scene']['yaxis'] = dict(
                showgrid=False,
                zeroline=False,
                showline=False,
                ticks='',
                showticklabels=False
            )

        if self.zaxis:
            plotly_object['layout']['scene']['zaxis'] = self.zaxis.to_plotly()
        elif not self.zaxis:
            plotly_object['layout']['scene']['zaxis'] = dict(
                showgrid=False,
                zeroline=False,
                showline=False,
                ticks='',
                showticklabels=False
            )

        return plotly_object
