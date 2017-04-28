from abc import ABCMeta, abstractmethod


class PlotlyWrapper(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def to_plotly(self):
        return {}
