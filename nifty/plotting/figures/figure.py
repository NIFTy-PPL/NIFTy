from nifty.plotting.plots.private import _Plot2D, _Plot3D
from figure_internal import _2dFigure, _3dFigure, _MapFigure, _BaseFigure
from nifty.plotting.plots import HeatMap, MollweideHeatmap
from nifty.plotting.figures.util import validate_plots
from plotly.tools import make_subplots


class Figure(_BaseFigure):
    def __init__(self, data, title=None, xaxis=None, yaxis=None, zaxis=None, width=None, height=None):
        _BaseFigure.__init__(self, data, title, width, height)
        kind, self.data = validate_plots(data)
        if kind == _Plot2D:
            if isinstance(self.data[0], HeatMap) and not width and not height:
                x = len(self.data[0].data)
                y = len(self.data[0].data[0])

                if x > y:
                    width = 500
                    height = int(500*y/x)
                else:
                    height = 500
                    width = int(500 * y / x)
                if isinstance(self.data[0], MollweideHeatmap):
                    if not xaxis:
                        xaxis = False
                    if not yaxis:
                        yaxis = False
            self.internal = _2dFigure(self.data, title, width, height, xaxis, yaxis)
        elif kind == _Plot3D:
            self.internal = _3dFigure(self.data, title, width, height, xaxis, yaxis, zaxis)
        elif kind:
            self.internal = _MapFigure(self.data, title)

    def _to_plotly(self):
        return self.internal._to_plotly()


class MultiFigure(_BaseFigure):
    def __init__(self, rows, cols, title=None, width=None, height=None):
        _BaseFigure.__init__(self, None, title, width, height)
        self.cols = cols
        self.rows = rows
        self.subfigures = []

    def get_subfigure(self, row, col):
        for fig, r, c, _, _ in self.subfigures:
            if r == row and c == col:
                return fig
        else:
            return None

    def add_subfigure(self, figure, row, col, row_span=1, col_span=1):
        self.subfigures.append((figure, row, col, row_span, col_span))

    def _to_plotly(self):
        sub_titles = tuple([a[0].title for a in self.subfigures])

        sub_specs = [[None]*self.cols for _ in range(self.rows)]
        for fig, r, c, rs, cs in self.subfigures:
            sub_specs[r][c] = dict(colspan=cs, rowspan=rs)
            if isinstance(fig.internal, _3dFigure):
                sub_specs[r][c]['is_3d'] = True
        multi_figure_ply = make_subplots(self.rows,self.cols, subplot_titles=sub_titles, specs=sub_specs)

        for fig, r, c, _, _ in self.subfigures:
            for plot in fig.data:
                multi_figure_ply.append_trace(plot._to_plotly(), r+1, c+1)

        multi_figure_ply['layout'].update(height=self.height, width=self.width, title=self.title)
        return multi_figure_ply


    @staticmethod
    def from_figures_2cols(figures, title=None, width=None, height=None):
        multi_figure = MultiFigure((len(figures)+1)/2, 2, title, width, height)

        for i in range(0, len(figures), 2):
            multi_figure.add_subfigure(figures[i], i/2, 0)

        for i in range(1, len(figures), 2):
            multi_figure.add_subfigure(figures[i], i/2, 1)

        return multi_figure


