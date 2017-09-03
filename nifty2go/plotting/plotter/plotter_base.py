# -*- coding: utf-8 -*-

from builtins import str
from builtins import zip
from builtins import range
import abc
import os
import sys

import numpy as np

from ...spaces.space import Space
from ...field import Field
from ... import nifty_utilities as utilities

from ..figures import MultiFigure
from future.utils import with_metaclass

import plotly

if plotly is not None and 'IPython' in sys.modules:
    plotly.offline.init_notebook_mode()


class PlotterBase(with_metaclass(abc.ABCMeta, type('NewBase', (object,), {}))):
    def __init__(self, interactive=False, path='plot.html', title=""):
        self.interactive = interactive
        self.path = path

        self.plot = self._initialize_plot()
        self.figure = self._initialize_figure()
        self.multi_figure = self._initialize_multifigure()

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

    def __call__(self, fields, spaces=None, data_extractor=None, labels=None,
                 path=None, title=None):
        if isinstance(fields, Field):
            fields = [fields]
        elif not isinstance(fields, list):
            fields = list(fields)

        spaces = utilities.cast_axis_to_tuple(spaces, len(fields[0].domain))

        if spaces is None:
            spaces = tuple(range(len(fields[0].domain)))

        if len(spaces) != len(self.domain_classes):
            raise ValueError("Domain mismatch between input and plotter.")

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
        for slice_list in utilities.get_slice_list(data_list[0].shape,
                                                   axes):
            plots_list += \
                    [[self.plot.at(self._parse_data(current_data,
                                                    field,
                                                    spaces))
                      for (current_data, field) in zip(data_list, fields)]]

        figures = [self.figure.at(plots, title=title)
                   for plots in plots_list]

        self._finalize_figure(figures, path=path)

    def _get_data_from_field(self, field, spaces, data_extractor):
        for i, space_index in enumerate(spaces):
            if not isinstance(field.domain[space_index],
                              self.domain_classes[i]):
                raise AttributeError("Given space(s) of input field-domain do "
                                     "not match the plotters domain.")

        # TODO: add data_extractor functionality here
        data = field.val
        return data

    @abc.abstractmethod
    def _initialize_plot(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _initialize_figure(self):
        raise NotImplementedError

    def _initialize_multifigure(self):
        return MultiFigure(subfigures=None)

    def _finalize_figure(self, figures, path=None):
        if len(figures) > 1:
            rows = (len(figures) + 1)//2
            figure_array = np.empty((2*rows), dtype=np.object)
            figure_array[:len(figures)] = figures
            figure_array = figure_array.reshape((2, rows))

            final_figure = self.multi_figure(subfigures=figure_array)
        else:
            final_figure = figures[0]

        path = self.path if path is None else path
        plotly.offline.plot(final_figure.to_plotly(),
                            filename=path)
