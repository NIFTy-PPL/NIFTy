from abc import ABCMeta, abstractmethod

class _PlotlyWrapper:
    __metaclass__ = ABCMeta

    @abstractmethod
    def _to_plotly(self):
        pass