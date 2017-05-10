from abc import ABCMeta, abstractmethod
from nifty.nifty_meta import NiftyMeta


class PlotlyWrapper(object):
    __metaclass__ = NiftyMeta

    @abstractmethod
    def to_plotly(self):
        return {}
