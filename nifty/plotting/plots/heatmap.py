from nifty.plotting.plots.private import _PlotBase, _Plot2D


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
        if self.smoothing:
            ply_object['zsmooth'] = self.smoothing
        return ply_object