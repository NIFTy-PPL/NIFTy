# -*- coding: utf-8 -*-

from nifty.plotting.plotly_wrapper import PlotlyWrapper


class Axis(PlotlyWrapper):
    def __init__(self, text=None, font='', color='', log=False,
                 font_size=18, show_grid=True, visible=True):
        self.text = text
        self.font = font
        self.color = color
        self.log = log
        self.font_size = int(font_size)
        self.show_grid = show_grid
        self.visible = visible

    def to_plotly(self):
        ply_object = dict()
        if self.text:
            ply_object.update(dict(
                title=self.text,
                titlefont=dict(
                    family=self.font,
                    color=self.color,
                    size=self.font_size
                )
            ))
        if self.log:
            ply_object['type'] = 'log'
        if not self.show_grid:
            ply_object['showgrid'] = False
        ply_object['visible'] = self.visible
        ply_object['tickfont'] = {'size': self.font_size}
        return ply_object
