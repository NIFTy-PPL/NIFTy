# -*- coding: utf-8 -*-

from nifty.plotting.plotly_wrapper import PlotlyWrapper


class FigureBase(PlotlyWrapper):
    def __init__(self, title, width, height):
        self.title = title
        self.width = width
        self.height = height
