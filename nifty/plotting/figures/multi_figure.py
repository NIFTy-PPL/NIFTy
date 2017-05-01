# -*- coding: utf-8 -*-

import numpy as np
from nifty import dependency_injector as gdi
from figure_base import FigureBase
from figure_3D import Figure3D

plotly = gdi.get('plotly')


# TODO: add nice height and width defaults for multifigure
class MultiFigure(FigureBase):
    def __init__(self, rows, columns, title=None, width=None, height=None,
                 subfigures=None):
        if 'plotly' not in gdi:
            raise ImportError("The module plotly is needed but not available.")
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
        title_extractor = lambda z: z.title if z else ""
        sub_titles = tuple(np.vectorize(title_extractor)(self.subfigures.flatten()))

        specs_setter = lambda z: {'is_3d': True} if isinstance(z, Figure3D) else {}
        sub_specs = list(map(list, np.vectorize(specs_setter)(self.subfigures)))

        multi_figure_plotly_object = plotly.tools.make_subplots(
                                                   self.rows,
                                                   self.columns,
                                                   subplot_titles=sub_titles,
                                                   specs=sub_specs)
        #TODO resolve bug with titles and 3D subplots
        for index, fig in np.ndenumerate(self.subfigures):
            if fig:
                for plot in fig.plots:
                    multi_figure_plotly_object.append_trace(plot.to_plotly(),
                                                            index[0]+1,
                                                            index[1]+1)

        multi_figure_plotly_object['layout'].update(height=self.height,
                                                    width=self.width,
                                                    title=self.title)

        return multi_figure_plotly_object


    @staticmethod
    def from_figures_cols(figures, cols=2, title=None, width=None, height=None):
        multi_figure = MultiFigure((len(figures) + cols -1)/cols , cols, title, width, height)
        for i in range(len(figures)):
            multi_figure.add_subfigure(figures[i], i/cols, i%cols)

        return multi_figure


