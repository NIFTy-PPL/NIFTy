# -*- coding: utf-8 -*-

import abc
import os

import plotly
from plotly import tools
import plotly.offline as ply


from keepers import Loggable

from nifty.spaces.space import Space
from nifty.field_types.field_type import FieldType
import nifty.nifty_utilities as utilities

plotly.offline.init_notebook_mode()


class Plotter(Loggable, object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, interactive=False, path='.', stack_subplots=False,
                 color_scale):
        self.interactive = interactive
        self.path = path
        self.stack_subplots = stack_subplots
        self.color_scale = None
        self.title = 'uiae'

    @abc.abstractproperty
    def domain(self):
        return (Space,)

    @abc.abstractproperty
    def field_type(self):
        return (FieldType,)

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

    @property
    def stack_subplots(self):
        return self._stack_subplots

    @stack_subplots.setter
    def stack_subplots(self, stack_subplots):
        self._stack_subplots = bool(stack_subplots)

    @abc.abstractmethod
    def plot(self, field, spaces=None, types=None, data_preselector=None):
        # if fields is a list, create a new field with appended
        # field_type = field_array and copy individual parts into the new field

        spaces = utilities.cast_axis_to_tuple(spaces, len(field.domain))
        types = utilities.cast_axis_to_tuple(types, len(field.field_type))
        if field.domain[spaces] != self.domain:
            raise AttributeError("Given space(s) of input field-domain do not "
                                 "match the plotters domain.")
        if field.field_type[spaces] != self.field_type:
            raise AttributeError("Given field_type(s) of input field-domain "
                                 "do not match the plotters field_type.")

        # iterate over the individual slices in order to compose the figure
        # -> make a d2o.get_full_data() (for rank==0 only?)

        # add clipping

        # no_subplot
        result_figure = self._create_individual_plot(data)
        # non-trivial subplots
        result_figure = tools.make_subplots(cols=2, rows='total_iterator%2 + 1',
                                            subplot_titles='iterator_index')

        self._finalize_figure(result_figure)

    def _create_individual_plot(self, data):
        pass

    def _finalize_figure(self, figure):
        if self.interactive:
            ply.iplot(figure)
        else:
            # is there a use for ply.plot when one has no intereset in
            # saving a file?

            # -> check for different file types
            # -> store the file to disk (MPI awareness?)






