from nifty.plotting.plotly_wrapper import _PlotlyWrapper
from nifty.plotting.plots.private import _Plot2D, _Plot3D
from figure_internal import _2dFigure, _3dFigure, _MapFigure
from nifty.plotting.plots import HeatMap
from nifty.plotting.figures.util import validate_plots


class Figure(_PlotlyWrapper):
    def __init__(self, data, title=None, width=None, height=None, xaxis=None, yaxis=None, zaxis=None):
        kind, data = validate_plots(data)
        if kind == _Plot2D:
            if isinstance(data[0], HeatMap) and not width and not height:
                x = len(data[0].data)
                y = len(data[0].data[0])

                if x > y:
                    width = 1000
                    height = int(1000*y/x)
                else:
                    height = 1000
                    width = int(1000 * y / x)
            self.internal = _2dFigure(data, title, width, height, xaxis, yaxis)
        elif kind == _Plot3D:
            self.internal = _3dFigure(data, title, width, height, xaxis, yaxis, zaxis)
        else:
            self.internal = _MapFigure(data, title)

    def _to_plotly(self):
        return self.internal._to_plotly()

