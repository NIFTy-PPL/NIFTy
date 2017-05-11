from nifty.spaces import HPSpace

from nifty.plotting.figures import Figure2D
from nifty.plotting.plots import Mollweide
from .plotter import Plotter


class HealpixPlotter(Plotter):
    def __init__(self, interactive=False, path='.', title="", color_map=None):
        super(HealpixPlotter, self).__init__(interactive, path, title)
        self.color_map = color_map

    @property
    def domain_classes(self):
        return (HPSpace, )

    def _create_individual_figure(self, plots):
        return Figure2D(plots)

    def _create_individual_plot(self, data, plot_domain):
        result_plot = Mollweide(data=data,
                                color_map=self.color_map)
        return result_plot
