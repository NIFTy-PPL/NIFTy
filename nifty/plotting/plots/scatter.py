from nifty.plotting.plots.private import _Scatter2DBase
from nifty.plotting.plots.private import _Plot3D, _Plot2D

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


class Scatter3D(_Scatter2DBase, _Plot3D):
    def __init__(self, x, y, z, label='', line=None, marker=None):
        _Scatter2DBase.__init__(self, x, y, label, line, marker)
        self.z = z

    def _to_plotly(self):
        ply_object = _Scatter2DBase._to_plotly(self)
        ply_object['z'] = self.z
        ply_object['type'] = 'scatter3d'
        return ply_object
