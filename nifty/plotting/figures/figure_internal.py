from abc import ABCMeta, abstractmethod
from nifty.plotting.plotly_wrapper import _PlotlyWrapper


class _BaseFigure(_PlotlyWrapper):
    __metaclass__ = ABCMeta

    def __init__(self, data, title, width, height):
        self.data = data
        self.title = title
        self.width = width
        self.height = height

    @abstractmethod
    def _to_plotly(self):
        ply_object = dict(
            data=[plt._to_plotly() for plt in self.data],
            layout=dict(
                title=self.title,
                scene = dict(
                    aspectmode='cube'
                ),
                autosize=False,
                width=self.width,
                height=self.height,
            )
        )
        return ply_object


class _2dFigure(_BaseFigure):
    def __init__(self, data, title=None, width=None, height=None, xaxis=None, yaxis=None):
        _BaseFigure.__init__(self, data, title, width, height)
        self.xaxis = xaxis
        self.yaxis = yaxis

    def _to_plotly(self):
        ply_object = _BaseFigure._to_plotly(self)
        if self.xaxis or self.yaxis:
            ply_object['layout']['scene']['aspectratio'] = dict()
        if self.xaxis:
            ply_object['layout']['xaxis'] = self.xaxis._to_plotly()
        elif self.xaxis == False:
            ply_object['layout']['xaxis'] = dict(
                                                autorange=True,
                                                showgrid=False,
                                                zeroline=False,
                                                showline=False,
                                                autotick=True,
                                                ticks='',
                                                showticklabels=False
                                            )

        if self.yaxis:
            ply_object['layout']['yaxis'] = self.yaxis._to_plotly()
        elif self.yaxis == False:
            ply_object['layout']['yaxis'] = dict(showline=False)
        return ply_object


class _3dFigure(_2dFigure):
    def __init__(self, data, title=None, width=None, height=None, xaxis=None, yaxis=None, zaxis=None):
        _2dFigure.__init__(self, data, title, width, height, xaxis, yaxis)
        self.zaxis = zaxis

    def _to_plotly(self):
        ply_object = _BaseFigure._to_plotly(self)
        if self.xaxis or self.yaxis or self.zaxis:
            ply_object['layout']['scene']['aspectratio'] = dict()
        if self.xaxis:
            ply_object['layout']['scene']['xaxis'] = self.xaxis._to_plotly()
        elif self.xaxis == False:
            ply_object['layout']['scene']['xaxis'] = dict(showline=False)
        if self.yaxis:
            ply_object['layout']['scene']['yaxis'] = self.yaxis._to_plotly()
        elif self.yaxis == False:
            ply_object['layout']['scene']['yaxis'] = dict(showline=False)
        if self.zaxis:
            ply_object['layout']['scene']['zaxis'] = self.zaxis._to_plotly()
        elif self.zaxis == False:
            ply_object['layout']['scene']['zaxis'] = dict(showline=False)
        return ply_object


class _MapFigure(_BaseFigure):
    def __init__(self, data, title, width=None, height=None):
        _BaseFigure.__init__(self, data, title, width, height)

    def _to_plotly(self):
        ply_object = _BaseFigure._to_plotly(self)
        # print(ply_object, ply_object['layout'])
        # ply_object['layout']['geo'] = dict(
        #     projection=dict(type=self.data.projection),
        #     showcoastlines=False
        # )
        return ply_object
