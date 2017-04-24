from abc import ABCMeta, abstractmethod


class Plottable:
    __metaclass__ = ABCMeta

    @abstractmethod
    def plot(self):
        pass
