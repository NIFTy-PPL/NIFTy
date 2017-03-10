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
                 color_scale=None):
        self.interactive = interactive
        self.path = path
        self.stack_subplots = stack_subplots
        self.color_scale = color_scale
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

    def plot(self, field, spaces=None, types=None, slice=None):
        data = self._get_data_from_field(field, spaces, types, slice)
        figures = self._create_individual_plot(data)
        self._finalize_figure(figures)

    @abc.abstractmethod
    def _get_data_from_field(self, field, spaces=None, types=None, slice=None):
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
            # add
        return [1,2,3]

    def _create_individual_plot(self, data):
        pass

    def _finalize_figure(self, figure):
        pass
        # is there a use for ply.plot when one has no interest in
        # saving a file?

        # -> check for different file types
        # -> store the file to disk (MPI awareness?)




