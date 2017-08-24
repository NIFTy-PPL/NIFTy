# -*- coding: utf-8 -*-

from builtins import map
from builtins import str
import numpy as np
from ... import dependency_injector as gdi
from .figure_base import FigureBase
from .figure_3D import Figure3D

plotly = gdi.get('plotly')


# TODO: add nice height and width defaults for multifigure
class MultiFigure(FigureBase):
    def __init__(self, subfigures, title=None, width=None, height=None):
        if 'plotly' not in gdi:
            raise ImportError("The module plotly is needed but not available.")
        super(MultiFigure, self).__init__(title, width, height)
        if subfigures is not None:
            self.subfigures = np.asarray(subfigures, dtype=np.object)
            if len(self.subfigures.shape) != 2:
                raise ValueError("Subfigures must be a two-dimensional array.")


    def at(self, subfigures):
        return MultiFigure(subfigures=subfigures,
                           title=self.title,
                           width=self.width,
                           height=self.height)

    @property
    def rows(self):
        return self.subfigures.shape[0]

    @property
    def columns(self):
        return self.subfigures.shape[1]

    def to_plotly(self):
        title_extractor = lambda z: z.title if z else ""
        sub_titles = tuple(np.vectorize(title_extractor)(
                                                    self.subfigures.flatten()))

        specs_setter = lambda z: ({'is_3d': True}
                                  if isinstance(z, Figure3D) else {})
        sub_specs = list(map(list, np.vectorize(specs_setter)(
                                                             self.subfigures)))

        multi_figure_plotly_object = plotly.tools.make_subplots(
                                                   self.rows,
                                                   self.columns,
                                                   subplot_titles=sub_titles,
                                                   specs=sub_specs)

        multi_figure_plotly_object['layout'].update(height=self.height,
                                                    width=self.width,
                                                    title=self.title)

        # TODO resolve bug with titles and 3D subplots

        i = 1
        for index, fig in np.ndenumerate(self.subfigures):
            if fig:
                for plot in fig.plots:
                    multi_figure_plotly_object.append_trace(plot.to_plotly(),
                                                            index[0]+1,
                                                            index[1]+1)
                    if isinstance(fig, Figure3D):
                        scene = dict()
                        if fig.xaxis:
                            scene['xaxis'] = fig.xaxis.to_plotly()
                        if fig.yaxis:
                            scene['yaxis'] = fig.yaxis.to_plotly()
                        if fig.zaxis:
                            scene['zaxis'] = fig.zaxis.to_plotly()

                        multi_figure_plotly_object['layout']['scene'+str(i)] = scene
                    else:
                        if fig.xaxis:
                            multi_figure_plotly_object['layout']['xaxis'+str(i)] = fig.xaxis.to_plotly()
                        if fig.yaxis:
                            multi_figure_plotly_object['layout']['yaxis'+str(i)] = fig.yaxis.to_plotly()

                    i += 1

        return multi_figure_plotly_object
