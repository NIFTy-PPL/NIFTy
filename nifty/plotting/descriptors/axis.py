# -*- coding: utf-8 -*-

from nifty.plotting.plotly_wrapper import PlotlyWrapper


class Axis(PlotlyWrapper):
    def __init__(self, label=None, font='Balto', color='', log=False,
                 font_size=22, show_grid=True, visible=True):
        self.label = str(label) if label is not None else None
        self.font = font
        self.color = color
        self.log = log
        self.font_size = int(font_size)
        self.show_grid = show_grid
        self.visible = visible

    def to_plotly(self):
        ply_object = dict()
        if self.label:
            ply_object.update(dict(
                title=self.label,
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
        ply_object['tickfont'] = {'size': self.font_size,
                                  'family': self.font}
        ply_object['exponentformat'] = 'power'
#        ply_object['domain'] = {'0': '0.04',
#                                '1': '1'}
        return ply_object
