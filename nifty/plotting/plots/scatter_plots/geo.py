from nifty.plotting.plots.plot import Plot


class Geo(Plot):
    def __init__(self, lon, lat, label='', line=None, marker=None,
                 proj='mollweide'):
        """
        proj: mollweide or mercator
        """

        super.__init__(label, line, marker)
        self.lon = lon
        self.lat = lat
        self.projection = proj

    def _to_plotly(self):
        plotly_object = super(Geo, self).to_plotly()
        plotly_object['type'] = 'scattergeo'
        plotly_object['lon'] = self.lon
        plotly_object['lat'] = self.lat
        if self.line:
            plotly_object['mode'] = 'lines'
        return plotly_object
