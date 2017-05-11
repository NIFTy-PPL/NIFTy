# -*- coding: utf-8 -*-

import abc
import os

import numpy as np

from keepers import Loggable

from nifty.config import dependency_injector as gdi

from nifty.spaces.space import Space
from nifty.field import Field
import nifty.nifty_utilities as utilities

from nifty.plotting.figures import MultiFigure

plotly = gdi.get('plotly')

try:
    plotly.offline.init_notebook_mode()
except AttributeError:
    pass

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank


class Plotter(Loggable, object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, interactive=False, path='.', title=""):
        if 'plotly' not in gdi:
            raise ImportError("The module plotly is needed but not available.")
        self.interactive = interactive
        self.path = path
        self.title = str(title)

    @abc.abstractproperty
    def domain_classes(self):
        return (Space,)

    @property
    def interactive(self):
        return self._interactive

    @interactive.setter
    def interactive(self, interactive):
        self._interactive = bool(interactive)

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, new_path):
        self._path = os.path.normpath(new_path)

    def plot(self, fields, spaces=None,  data_extractor=None, labels=None):
        if isinstance(fields, Field):
            fields = [fields]
        elif not isinstance(fields, list):
            fields = list(fields)

        spaces = utilities.cast_axis_to_tuple(spaces, len(fields[0].domain))

        if spaces is None:
            spaces = tuple(range(len(fields[0].domain)))

        axes = []
        plot_domain = []
        for space_index in spaces:
            axes += list(fields[0].domain_axes[space_index])
            plot_domain += [fields[0].domain[space_index]]

        # prepare data
        data_list = [self._get_data_from_field(field, spaces, data_extractor)
                     for field in fields]

        # create plots
        plots_list = []
        for slice_list in utilities.get_slice_list(data_list[0].shape, axes):
            plots_list += \
                    [[self._create_individual_plot(current_data[slice_list],
                                                   plot_domain)
                      for current_data in data_list]]

        figures = [self._create_individual_figure(plots)
                   for plots in plots_list]

        self._finalize_figure(figures)

    def _get_data_from_field(self, field, spaces, data_extractor):
        for i, space_index in enumerate(spaces):
            if not isinstance(field.domain[space_index],
                              self.domain_classes[i]):
                raise AttributeError("Given space(s) of input field-domain do "
                                     "not match the plotters domain.")

        # TODO: add data_extractor functionality here
        data = field.val.get_full_data(target_rank=0)
        return data

    @abc.abstractmethod
    def _create_individual_figure(self, plots):
        raise NotImplementedError

    @abc.abstractmethod
    def _create_individual_plot(self, data, fields):
        raise NotImplementedError

    def _finalize_figure(self, figures):
        if len(figures) > 1:
            rows = (len(figures) + 1)//2
            figure_array = np.empty((2*rows), dtype=np.object)
            figure_array[:len(figures)] = figures
            figure_array = figure_array.reshape((2, rows))

            final_figure = MultiFigure(rows, 2,
                                       title='Test',
                                       subfigures=figure_array)
        else:
            final_figure = figures[0]

        plotly.offline.plot(final_figure.to_plotly(),
                            filename=os.path.join(self.path, self.title))
