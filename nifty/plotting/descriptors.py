from plotly_wrapper import _PlotlyWrapper


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


class Axis(_PlotlyWrapper):
    def __init__(self, text=None, font='', color='', log=False, aspect_ratio=None):
        self.text = text
        self.font = font
        self.color = color
        self.log = log
        self.aspect_ratio = aspect_ratio

    def _to_plotly(self):
        ply_object = dict()
        if self.text:
            ply_object.update(dict(
                title=self.text,
                titlefont=dict(
                    family=self.font,
                    color=self.color
                )
            ))
        if self.log:
            ply_object['type'] = 'log'

        return ply_object
