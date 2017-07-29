from nifty.spaces import HPSpace

from nifty.plotting.figures import Figure2D
from nifty.plotting.plots import HPMollweide
from .plotter_base import PlotterBase


class HealpixPlotter(PlotterBase):
    def __init__(self, interactive=False, path='plot.html', color_map=None):
        self.color_map = color_map
        super(HealpixPlotter, self).__init__(interactive, path)

    @property
    def domain_classes(self):
        return (HPSpace, )

    def _initialize_plot(self):
        result_plot = HPMollweide(data=None,
                                  color_map=self.color_map)
        return result_plot

    def _initialize_figure(self):
        return Figure2D(plots=None)

    def _parse_data(self, data, field, spaces):
        return data
