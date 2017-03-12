from abc import ABCMeta, abstractmethod
from nifty.plotting.plotly_wrapper import _PlotlyWrapper
from nifty.plotting.descriptors import Marker

class _PlotBase(_PlotlyWrapper):
    __metaclass__ = ABCMeta

    def __init__(self, label, line, marker):
        self.label = label
        self.line = line
        self.marker = marker
        if not line and not marker:
            self.marker = Marker()

    @abstractmethod
    def _to_plotly(self):
        ply_object = dict()
        ply_object['name'] = self.label
        if self.line and self.marker:
            ply_object['mode'] = 'lines+markers'
            ply_object['line'] = self.line._to_plotly()
            ply_object['marker'] = self.marker._to_plotly()
        elif self.line:
            ply_object['mode'] = 'line'
            ply_object['line'] = self.line._to_plotly()
        elif self.marker:
            ply_object['mode'] = 'markers'
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
