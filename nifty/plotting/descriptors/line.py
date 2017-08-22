# -*- coding: utf-8 -*-

from ..plotly_wrapper import PlotlyWrapper


class Line(PlotlyWrapper):
    def __init__(self, color=None, width=None):
        self.color = color
        self.width = width

    def to_plotly(self):
        return dict(color=self.color,
                    width=self.width)
