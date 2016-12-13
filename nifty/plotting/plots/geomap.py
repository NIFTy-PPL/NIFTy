from nifty.plotting.plots.private import _PlotBase


class ScatterGeoMap(_PlotBase):
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
