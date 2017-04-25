# -*- coding: utf-8 -*-

import numpy as np

from plotly.tools import make_subplots

from figure_base import FigureBase
from figure_3D import Figure3D


class MultiFigure(FigureBase):
    def __init__(self, rows, columns, subfigures=None,
                 title=None, width=None, height=None):
        super(MultiFigure, self).__init__(title, width, height)
        self.subfigures = np.empty((rows, columns), dtype=np.object)
        self.subfigures[:] = subfigures

    @property
    def rows(self):
        return self.subfigures.shape[0]

    @property
    def columns(self):
        return self.subfigures.shape[1]

    def add_subfigure(self, figure, row, column):
        self.subfigures[row, column] = figure

    def to_plotly(self):
        sub_titles = self.subfigures.copy()
        sub_titles = sub_titles.flatten
        title_extractor = lambda z: z.title
        sub_titles = np.vectorize(title_extractor)(sub_titles)

        sub_specs = self.subfigures.copy_empty()
        specs_setter = \
            lambda z: {'is_3d': True} if isinstance(z, Figure3D) else {}
        sub_specs = np.vectorize(specs_setter)(sub_specs)
        multi_figure_plotly_object = make_subplots(self.rows,
                                                   self.columns,
                                                   subplot_titles=sub_titles,
                                                   specs=sub_specs)

        for index, fig in np.ndenumerate(self.subfigures):
            for plot in fig.plots:
                multi_figure_plotly_object.append_trace(plot.to_plotly(),
                                                        index[0]+1,
                                                        index[1]+1)

        multi_figure_plotly_object['layout'].update(height=self.height,
                                                    width=self.width,
                                                    title=self.title)
        return multi_figure_plotly_object


    @staticmethod
    def from_figures_2cols(figures, title=None, width=None, height=None):
        multi_figure = MultiFigure((len(figures)+1)/2, 2, title, width, height)

        for i in range(0, len(figures), 2):
            multi_figure.add_subfigure(figures[i], i/2, 0)

        for i in range(1, len(figures), 2):
            multi_figure.add_subfigure(figures[i], i/2, 1)

        return multi_figure


