from nifty.plotting.plotly_wrapper import _PlotlyWrapper
from nifty.plotting.plots.private import _Plot2D, _Plot3D
from private import _2dFigure, _3dFigure, _MapFigure
from nifty.plotting.plots import ScatterGeoMap, HeatMap



class Figure(_PlotlyWrapper):
    def __init__(self, data, title=None, width=None, height=None, xaxis=None, yaxis=None, zaxis=None):
        if not data:
            raise Exception('Error: no plots given')

        if type(data) != list:
            raise Exception('Error: plots should be passed in a list')

        if isinstance(data[0], _Plot2D):
            kind = _Plot2D
        elif isinstance(data[0], _Plot3D):
            kind = _Plot3D
        elif isinstance(data[0], ScatterGeoMap):
            kind = ScatterGeoMap
        else:
            kind = None

        if kind:
            for plt in data:
                if not isinstance(plt, kind):
                    raise Exception(
                        """Error: Plots are not of the right kind!
                        Compatible types are:
                         - Scatter2D and HeatMap
                         - Scatter3D
                         - ScatterMap""")
        else:
            raise Exception('Error: plot type unknown')

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
