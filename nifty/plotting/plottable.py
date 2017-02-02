from abc import ABCMeta, abstractmethod


class Plottable(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def plot(self):
        pass
