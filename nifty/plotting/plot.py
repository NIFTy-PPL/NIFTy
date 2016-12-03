from abc import ABCMeta, abstractmethod
import os

import time
t1 = time.time()
import plotly.offline as ply_offline
import plotly.plotly as ply
from plotly import tools
t2 = time.time()

print ('import time', t2-t1)


class _PlotlyWrapper:
    __metaclass__ = ABCMeta

    @abstractmethod
    def _to_plotly(self):
        pass


class Marker(_PlotlyWrapper):
    # find symbols at: https://plot.ly/python/reference/#scatter-marker-symbol
    def __init__(self, color=None, size=None, symbol=None, opacity=None):
        self.color = color
        self.size = size
        self.symbol = symbol
        self.opacity = opacity

    def _to_plotly(self):
        return dict(color=self.color, size=self.size, symbol=self.symbol, opacity=self.opacity)


class Line(_PlotlyWrapper):
    def __init__(self, color=None, width=None):
        self.color = color
        self.width = width

    def _to_plotly(self):
        return dict(color=self.color, width=self.width)


class _PlotBase(_PlotlyWrapper):
    __metaclass__ = ABCMeta

    def __init__(self, label, line, marker):
        self.label = label
        self.line = line
        self.marker = marker

    @abstractmethod
    def _to_plotly(self):
        ply_object = dict()
        ply_object['name'] = self.label
        if self.line and self.marker:
            ply_object['mode'] = 'lines+markers'
            ply_object['line'] = self.line._to_plotly()
            ply_object['marker'] = self.marker._to_plotly()
        elif self.line:
            ply_object['mode'] = 'markers'
            ply_object['line'] = self.line._to_plotly()
        elif self.marker:
            ply_object['mode'] = 'line'
            ply_object['marker'] = self.marker._to_plotly()

        return ply_object


class _Scatter2DBase(_PlotBase):
    __metaclass__ = ABCMeta

    def __init__(self, x, y, label, line, marker):
        _PlotBase.__init__(self, label, line, marker)
        self.x = x
        self.y = y

    @abstractmethod
    def _to_plotly(self):
        ply_object = _PlotBase._to_plotly(self)
        ply_object['x'] = self.x
        ply_object['y'] = self.y

        return ply_object

class _Plot2D:
    pass # only used as a labeling system for the plots represntation

class _Plot3D:
    pass # only used as a labeling system for the plots represntation


class Scatter2D(_Scatter2DBase, _Plot2D):
    def __init__(self, x=None, y=None, x_start=0, x_step=1,
                 label='', line=None, marker=None, webgl=True):
        if y is None:
            raise Exception('Error: no y data to plot')
        if x is None:
            x = range(x_start, len(y) * x_step, x_step)
        _Scatter2DBase.__init__(self, x, y, label, line, marker)
        self.webgl = webgl


    def _to_plotly(self):
        ply_object = _Scatter2DBase._to_plotly(self)
        if self.webgl:
            ply_object['type'] = 'scattergl'
        else:
            ply_object['type'] = 'scatter'

        return ply_object


class Scatter3D(_Scatter2DBase, _Plot2D):
    def __init__(self, x, y, z, label='', line=None, marker=None):
        _Scatter2DBase.__init__(self, x, y, label, line, marker)
        self.z = z

    def _to_plotly(self):
        ply_object = _Scatter2DBase._to_plotly(self)
        ply_object['z'] = self.z
        ply_object['type'] = 'scatter3d'
        return ply_object


class ScatterMap(_PlotBase):
    def __init__(self, lon, lat, label='', line=None, marker=None, proj='mollweide'): # or  'mercator'
        _PlotBase.__init__(self, label, line, marker)
        self.lon = lon
        self.lat = lat
        self.projection = proj

    def _to_plotly(self):
        ply_object = _PlotBase._to_plotly(self)
        ply_object['type'] = 'scattergeo'
        ply_object['lon'] = self.lon
        ply_object['lat'] = self.lat
        if self.line:
            ply_object['mode'] = 'lines'
        return ply_object


class HeatMap(_PlotBase, _Plot2D):
    def __init__(self, data, label='', line=None, marker=None, webgl=False, smoothing=False): # smoothing 'best', 'fast', False
        _PlotBase.__init__(self, label, line, marker)
        self.data = data
        self.webgl = webgl
        self.smoothing = smoothing


    def _to_plotly(self):
        ply_object = _PlotBase._to_plotly(self)
        ply_object['z'] = self.data
        if self.webgl:
            ply_object['type'] = 'heatmapgl'
        else:
            ply_object['type'] = 'heatmap'
        ply_object['zsmooth'] = self.smoothing
        return ply_object

class Axis(_PlotlyWrapper):
    def __init__(self, text, font='', color='', log=False):
        self.text = text
        self.font = font
        self.color = color
        self.log = log

    def _to_plotly(self):
        ply_object = dict(
            title=self.text,
            titlefont=dict(
                family=self.font,
                color=self.color
            )
        )
        if self.log:
            ply_object['type'] = 'log'

        return ply_object


class Figure(_PlotlyWrapper):
    class _BaseFigure(_PlotlyWrapper):
        __metaclass__ = ABCMeta
        def __init__(self, data, title):
            self.data = data
            self.title = title

        @abstractmethod
        def _to_plotly(self):
            ply_object = dict(data=[plt._to_plotly() for plt in self.data], layout=dict(title=self.title))
            return ply_object

    class _2dFigure(_BaseFigure):
        def __init__(self, data, title=None, xaxis=None, yaxis=None):
            Figure._BaseFigure.__init__(self, data, title)
            self.xaxis = xaxis
            self.yaxis = yaxis

        def _to_plotly(self):
            ply_object = Figure._BaseFigure._to_plotly(self)
            if self.xaxis:
                ply_object['layout']['xaxis'] = self.xaxis._to_plotly()
            if self.yaxis:
                ply_object['layout']['yaxis'] = self.yaxis._to_plotly()
            return ply_object

    class _3dFigure(_2dFigure):
        def __init__(self, data, title=None, xaxis=None, yaxis=None, zaxis=None):
            Figure._2dFigure.__init__(self, data, title, xaxis, yaxis)
            self.zaxis=zaxis

        def _to_plotly(self):
            ply_object = Figure._BaseFigure._to_plotly(self)
            ply_object['layout']['scene'] = dict()
            if self.xaxis:
                ply_object['layout']['scene']['xaxis'] = self.xaxis._to_plotly()
            if self.yaxis:
                ply_object['layout']['scene']['yaxis'] = self.yaxis._to_plotly()
            if self.zaxis:
                ply_object['layout']['scene']['zaxis'] = self.zaxis._to_plotly()
            return ply_object

    class _MapFigure(_BaseFigure):
        def __init__(self, data, title):
            Figure._BaseFigure.__init__(self, data, title)

        def _to_plotly(self):
            ply_object = Figure._BaseFigure._to_plotly(self)
            # print(ply_object, ply_object['layout'])
            # ply_object['layout']['geo'] = dict(
            #     projection=dict(type=self.data.projection),
            #     showcoastlines=False
            # )
            return ply_object

    def __init__(self, data, title=None, xaxis=None, yaxis=None, zaxis=None):
        if not data:
            raise Exception('Error: no plots given')

        if type(data) != list:
            raise Exception('Error: plots should be passed in a list')

        if isinstance(data[0], _Plot2D):
            kind = _Plot2D
        elif isinstance(data[0], _Plot3D):
            kind = _Plot3D
        elif isinstance(data[0], ScatterMap):
            kind = ScatterMap
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
            self.internal = Figure._2dFigure(data, title, xaxis, yaxis)
        elif kind == _Plot3D:
            self.internal = Figure._3dFigure(data, title, xaxis, yaxis, zaxis)
        else:
            self.internal = Figure._MapFigure(data, title)

    def _to_plotly(self):
        return self.internal._to_plotly()


def plot(figure, filename=None):
    if not filename:
        filename = os.path.abspath('/tmp/temp-plot.html')
    ply_offline.plot(figure._to_plotly(), filename=filename)


def plot_image(figure, filename=None, show=False):
    try:
        if not filename:
            filename = os.path.abspath('temp-plot.jpeg')
        ply.image.save_as(figure._to_plotly(), filename=filename)
        if show:
            ply.image.ishow(figure._to_plotly())
    except:
        raise Exception('Error: Invalid image format! Try: png, svg, jpeg, or pdf')
# f = tools.make_subplots(rows = 2, cols=1)

# f.append_figure


# test
#
# import numpy as np
#
# N = 1000
# x = np.random.randn(N)
# y = np.random.randn(N)
# z = np.random.randn(N)
#
# #
# # data = [[1,2,3], [1,2,3], [1,2,3]]
# # h = HeatMap(data)
# # plot(Figure([h]))
# #
# # s3 = Scatter3D(x, y, z)
# # plot(Figure([s3]))
#
# s = Scatter2D(y=y, label='this is label')
# s2 = Scatter2D(x=x, y=x, marker=Marker(color='red'), line=Line(width=10))
# fig = Figure([s,s2], title='PAM', xaxis=Axis('I AM X', color='blue'))
#
# plot(fig)

