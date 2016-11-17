from abc import ABCMeta, abstractmethod

import plotly.graph_objs as go
import plotly.offline as ply

class GraphicsBase:
    def __init__(self, graphics):
        self.graphics = graphics


class Marker(GraphicsBase):
    # find symbols at: https://plot.ly/python/reference/#scatter-marker-symbol
    def __init__(self, color=None, size=None, symbol=None, opacity=None):
        GraphicsBase.__init__(self, dict(color=color, size=size, symbol=symbol, opacity=opacity))


class Line(GraphicsBase):
    def __init__(self, color=None, width=None):
        GraphicsBase.__init__(self, dict(color=color, width=width))


class PlotBase:
    __metaclass__ = ABCMeta

    def __init__(self, label, line, marker):
        self.label = label
        self.line = line
        self.marker = marker
        self.plotly = None

    @abstractmethod
    def _init_plotly(self):
        if self.line:
            self.plotly.mode = 'line'
            self.plotly.line = self.line.graphics
        if self.marker:
            self.plotly.mode = 'markers'
            self.plotly.marker = self.marker.graphics

        if self.line and self.marker:
            self.plotly.mode = 'lines+markers'


class Scatter2DBase(PlotBase):
    def __init__(self, x, y, label, line, marker):
        PlotBase.__init__(self, label, line, marker)
        self.x = x
        self.y = y
        self.PlyScaterType = None

    def _init_plotly(self):
        self.plotly = self.PlyScaterType (
            x=self.x,
            y=self.y
        )
        PlotBase._init_plotly(self)


class Scatter2D(Scatter2DBase):
    def __init__(self, x, y, label='', line=None, marker=None, webgl=True):
        Scatter2DBase.__init__(self, x, y, label, line, marker)
        self.webgl = webgl
        if self.webgl:
            self.PlyScaterType = go.Scattergl
        else:
            self.PlyScaterType = go.Scatter

        self._init_plotly()


class Scatter1D(Scatter2D):
    def __init__(self, y, x_start=0, x_step=1, label='', line=None, marker=None, webgl=True):
        Scatter2D.__init__(self, range(x_start, len(y) * x_step, x_step), y, label, line, marker, webgl)
        self.xStart = x_start
        self.xStep = x_step


class Scatter3D(Scatter2DBase):
    def __init__(self, x, y, z, label='', line=None, marker=None):
        Scatter2DBase.__init__(self, x, y, label, line, marker)
        self.z = z
        self.PlyScaterType = go.Scatter3d
        self._init_plotly()

    def _init_plotly(self):
        Scatter2DBase._init_plotly(self)
        self.plotly.z = self.z


# class Map(PlotBase):
#     def __init__(self, lon, lat, label='', line=None, marker=None, proj='mollweide'): # or  'mercator'
#         PlotBase.__init__(self, label, line, marker)
#         self.lon = lon
#         self.lat = lat
#         self.projection = proj
#         self.PlyScaterType = go.Scattergeo
#
#     def _init_plotly(self):
#         self.plotly = self.PlyScaterType(
#             lon=self.lon,
#             lat=self.lat
#         )
#         PlotBase._init_plotly(self)
#
#
# class HeatMap(PlotBase):
#     pass


def plot(data):
    ply.plot([plt.plotly for plt in data])

import numpy as np

N = 100
x = np.random.randn(N)
y = np.random.randn(N)
z = np.random.randn(N)


s = Scatter1D(y=y, marker=Marker(color='red'), line=Line(width=10))

plot([s])
