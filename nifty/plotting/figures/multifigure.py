from nifty.plotting.plotly_wrapper import _PlotlyWrapper
from nifty.plotting.figures.util import validate_plots


class MultiFigure(_PlotlyWrapper):
    def __init__(self, cols, rows, title=None, width=None, height=None):
        self.cols = cols
        self.rows = rows
        self.title = title
        self.width = width
        self.height = height

    def addSubfigure(self, data, title=None, width=None, height=None):
        kind, data = validate_plots(data)

    def _to_plotly(self):
        pass

