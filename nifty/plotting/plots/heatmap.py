from nifty.plotting.plots.private import _Plot2D
from nifty.plotting.plotly_wrapper import _PlotlyWrapper

import healpy.projaxes as PA
import healpy.pixelfunc as pixelfunc


class HeatMap(_PlotlyWrapper, _Plot2D):
    def __init__(self, data, webgl=False,
                 smoothing=False):  # smoothing 'best', 'fast', False
        self.data = data
        self.webgl = webgl
        self.smoothing = smoothing

    def _to_plotly(self):
        ply_object = dict()
        ply_object['z'] = self.data
        if self.webgl:
            ply_object['type'] = 'heatmapgl'
        else:
            ply_object['type'] = 'heatmap'
        if self.smoothing:
            ply_object['zsmooth'] = self.smoothing
        return ply_object


class MollweideHeatmap(HeatMap):
    def __init__(self, data, webgl=False,
                 smoothing=False):  # smoothing 'best', 'fast', False
        HeatMap.__init__(self, _mollview(data), webgl, smoothing)


def _mollview(x, xsize=800):
    import pylab
    x = pixelfunc.ma_to_array(x)
    f = pylab.figure(None, figsize=(8.5, 5.4))
    extent = (0.02, 0.05, 0.96, 0.9)
    ax = PA.HpxMollweideAxes(f, extent)
    img = ax.projmap(x, nest=False, xsize=xsize)
    return img
