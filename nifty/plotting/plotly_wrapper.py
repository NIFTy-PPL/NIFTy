from builtins import object
from abc import ABCMeta, abstractmethod
from nifty.nifty_meta import NiftyMeta
from future.utils import with_metaclass


class PlotlyWrapper(with_metaclass(NiftyMeta, object)):
    @abstractmethod
    def to_plotly(self):
        return {}
