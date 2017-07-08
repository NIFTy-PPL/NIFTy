# -*- coding: utf-8 -*-

from nifty import Energy


class QuadraticPotential(Energy):
    def __init__(self, position, eigenvalues):
        super(QuadraticPotential, self).__init__(position)
        self.eigenvalues = eigenvalues

    def at(self, position):
        return self.__class__(position,
                              eigenvalues=self.eigenvalues)

    @property
    def value(self):
        H = 0.5 * self.position.vdot(
                    self.eigenvalues(self.position))
        return H.real

    @property
    def gradient(self):
        g = self.eigenvalues(self.position)
        return g

    @property
    def curvature(self):
        return self.eigenvalues
