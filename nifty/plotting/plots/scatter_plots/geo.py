from .scatter_plot import ScatterPlot


class Geo(ScatterPlot):
    def __init__(self, data, label='', line=None, marker=None,
                 projection='mollweide'):
        """
        proj: mollweide or mercator
        """

        super(Geo, self).__init__(label, line, marker)
        self.projection = projection

    def at(self, data):
        return Geo(data=data,
                   label=self.label,
                   line=self.line,
                   marker=self.marker,
                   projection=self.projection)

    @property
    def figure_dimension(self):
        return 2

    def _to_plotly(self, data):
        plotly_object = super(Geo, self).to_plotly()
        plotly_object['type'] = self.projection
        plotly_object['lon'] = data[0]
        plotly_object['lat'] = data[1]
        if self.line:
            plotly_object['mode'] = 'lines'
        return plotly_object
