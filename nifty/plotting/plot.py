from abc import ABCMeta, abstractmethod
import os

import time
t1 = time.time()
import plotly.graph_objs as go
import plotly.offline as ply
t2 = time.time()

print ('import time', t2-t1)


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


class _PlotBase:
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

    @classmethod
    def plot(cls, plots, filename=None, title=''):
        if not filename:
            filename = os.path.abspath('/tmp/temp-plot.html')
        if plots:
            for plt in plots:
                if type(plt) is not cls:
                    raise Exception('Error: Plots are not of the desired type')
        else:
            print 'Warning: no plots given'
        fig = dict(data=[plt.plotly for plt in plots],layout=dict(title=title))
        ply.plot(fig, validate=False, filename=filename)


class _Scatter2DBase(_PlotBase):
    def __init__(self, x, y, label, line, marker):
        _PlotBase.__init__(self, label, line, marker)
        self.x = x
        self.y = y
        self.PlyScaterType = None

    def _init_plotly(self):
        self.plotly = self.PlyScaterType (
            x=self.x,
            y=self.y
        )
        _PlotBase._init_plotly(self)


class Scatter2D(_Scatter2DBase):
    def __init__(self, x=None, y=None, x_start=0, x_step=1,
                 label='', line=None, marker=None, webgl=True):
        if y is None:
            raise Exception('Error: no y data to plot')
        if x is None:
            x = range(x_start, len(y) * x_step, x_step)
        _Scatter2DBase.__init__(self, x, y, label, line, marker)
        self.webgl = webgl
        if self.webgl:
            self.PlyScaterType = go.Scattergl
        else:
            self.PlyScaterType = go.Scatter
        self._init_plotly()

    @classmethod
    def plot(cls, plots, filename=None, title='',
             xaxis_label=None, xaxis_log=False, yaxis_label=None, yaxis_log=False):
        if not filename:
            filename = os.path.abspath('/tmp/temp-plot.html')
        if plots:
            for plt in plots:
                if type(plt) is not cls:
                    raise Exception('Error: Plots are not of the desired type')
        else:
            print 'Warning: no plots given'
        fig = dict(data=[plt.plotly for plt in plots],
                   layout=dict(
                       title=title,
                       xaxis=dict(
                           title=xaxis_label,
                           type='log' if xaxis_log else '',
                           autoarange=xaxis_log
                       ) if xaxis_label else '',
                       yaxis=dict(
                           title=yaxis_label,
                           type='log' if yaxis_log else '',
                           autoarange=yaxis_log
                       ) if yaxis_label else ''
                   ))
        ply.plot(fig, validate=False, filename=filename)


class Scatter3D(_Scatter2DBase):
    def __init__(self, x, y, z, label='', line=None, marker=None):
        _Scatter2DBase.__init__(self, x, y, label, line, marker)
        self.z = z
        self.PlyScaterType = go.Scatter3d
        self._init_plotly()

    def _init_plotly(self):
        _Scatter2DBase._init_plotly(self)
        self.plotly.z = self.z


class ScatterMap(_PlotBase):
    def __init__(self, lon, lat, label='', line=None, marker=None, proj='mollweide'): # or  'mercator'
        _PlotBase.__init__(self, label, line, marker)
        self.lon = lon
        self.lat = lat
        self.projection = proj
        self.PlyScaterType = go.Scattergeo
        self._init_plotly()


    def _init_plotly(self):
        self.plotly = self.PlyScaterType(
            lon=self.lon,
            lat=self.lat
        )
        _PlotBase._init_plotly(self)
        if self.line:
            self.plotly.mode = 'lines'


    @classmethod
    def plot(cls, plots, filename=None, title=''):
        if not filename:
            filename = os.path.abspath('/tmp/temp-plot.html')
        if plots:
            for plt in plots:
                if type(plt) is not cls:
                    raise Exception('Error: Plots are not of the desired type')
        else:
            print 'Warning: no plots given'
            return

        fig = go.Figure(data=[plt.plotly for plt in plots],
                   layout=dict(
                    title=title,
                    height=800,
                    geo=dict(
                        projection=dict(type=plots[0].projection),
                        showcoastlines=False
                    )))

        ply.plot(fig, validate=False, filename=filename)


class HeatMap(_PlotBase):
    def __init__(self, data, label='', line=None, marker=None, webgl=False, smoothing=False): # smoothing 'best', 'fast', False
        self.data = data
        _PlotBase.__init__(self, label, line, marker)
        self.webgl = webgl
        if self.webgl:
            self.PlyScaterType = 'heatmapgl'
        else:
            self.PlyScaterType = 'heatmap'
        self.smoothing=smoothing
        self._init_plotly()

    def _init_plotly(self):
        self.plotly = dict(
            type=self.PlyScaterType,
            z=self.data,
            zsmooth=self.smoothing
        )
        _PlotBase._init_plotly(self)

    @classmethod
    def plot(cls, plot, filename=None, title=''):
        if not filename:
            filename = os.path.abspath('/tmp/temp-plot.html')
        if not plot:
            print 'Warning: no plots given'

        fig = dict(data=[plot.plotly], layout=dict(title=title))
        ply.plot(fig, validate=False, filename=filename)


#test

import numpy as np

N = 1000
x = np.random.randn(N)
y = np.random.randn(N)
z = np.random.randn(N)

s = Scatter2D(y=y, label='this is label')
s2 = Scatter2D(x=x, y=x, marker=Marker(color='red'), line=Line(width=10))

Scatter2D.plot([s],'scatter',
               title='THIS IS TITLE',
               xaxis_label='xaxis',
               yaxis_label='yaxis',
               xaxis_log=True)


data = [[1,2,3], [1,2,3], [1,2,3]]
#
# HeatMap.plot(HeatMap(data, webgl=True))
#
# geo = ScatterMap(x*90, y*360-180)
# ScatterMap.plot([geo], 'map')

