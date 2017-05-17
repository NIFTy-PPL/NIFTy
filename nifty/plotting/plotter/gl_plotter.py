from nifty.spaces import GLSpace

from nifty.plotting.figures import Figure2D
from nifty.plotting.plots import GLMollweide
from .plotter import Plotter


class GLPlotter(Plotter):
    def __init__(self, interactive=False, path='.', title="", color_map=None):
        super(GLPlotter, self).__init__(interactive, path, title)
        self.color_map = color_map

    @property
    def domain_classes(self):
        return (GLSpace, )

    def _create_individual_figure(self, plots):
        return Figure2D(plots)

    def _create_individual_plot(self, data, plot_domain):
        result_plot = GLMollweide(data=data, nlat=plot_domain[0].nlat,
                                  nlon=plot_domain[0].nlon,
                                  color_map=self.color_map)
        return result_plot
