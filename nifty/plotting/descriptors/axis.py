# -*- coding: utf-8 -*-

from nifty.plotting.plotly_wrapper import PlotlyWrapper


class Axis(PlotlyWrapper):
    def __init__(self, text=None, font='', color='', log=False,
                 show_grid=True):
        self.text = text
        self.font = font
        self.color = color
        self.log = log
        self.show_grid = show_grid

    def to_plotly(self):
        ply_object = dict()
        if self.text:
            ply_object.update(dict(
                title=self.text,
                titlefont=dict(
                    family=self.font,
                    color=self.color
                )
            ))
        if self.log:
            ply_object['type'] = 'log'
        if not self.show_grid:
            ply_object['showgrid'] = False
        return ply_object
