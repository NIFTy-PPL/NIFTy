
import numpy as np

from nifty.spaces import GLSpace

from nifty.plotting.figures import Figure2D
from nifty.plotting.plots import GLMollweide
from .plotter_base import PlotterBase


class GLPlotter(PlotterBase):
    def __init__(self, interactive=False, path='.', title="", color_map=None):
        self.color_map = color_map
        super(GLPlotter, self).__init__(interactive, path, title)

    @property
    def domain_classes(self):
        return (GLSpace, )

    def _initialize_plot(self):
        result_plot = GLMollweide(data=None,
                                  color_map=self.color_map)
        return result_plot

    def _initialize_figure(self):
        return Figure2D(plots=None)

    def _parse_data(self, data, field, spaces):
        gl_space = field.domain[spaces[0]]
        data = np.reshape(data, (gl_space.nlat, gl_space.nlon))
        return data
