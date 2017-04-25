# -*- coding: utf-8 -*-

from nifty.plotting.plotly_wrapper import PlotlyWrapper


class Marker(PlotlyWrapper):
    # find symbols at: https://plot.ly/python/reference/#scatter-marker-symbol
    def __init__(self, color=None, size=None, symbol=None, opacity=None):
        self.color = color
        self.size = size
        self.symbol = symbol
        self.opacity = opacity

    def to_plotly(self):
        return dict(color=self.color,
                    size=self.size,
                    symbol=self.symbol,
                    opacity=self.opacity)
