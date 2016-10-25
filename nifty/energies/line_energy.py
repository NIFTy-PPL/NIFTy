# -*- coding: utf-8 -*-

from .energy import Energy


class LineEnergy(Energy):
    def __init__(self, position, energy, line_direction):
        self.energy = energy
        self.line_direction = line_direction
        super(LineEnergy, self).__init__(position=position)

    def at(self, position):
        if position == 0:
            return self
        else:
            full_position = self.position + self.line_direction*position
            return self.__class__(full_position,
                                  self.energy,
                                  self.line_direction)

    @property
    def value(self):
        return self.energy.value

    @property
    def gradient(self):
        return self.energy.gradient.dot(self.line_direction)

    @property
    def curvature(self):
        return self.energy.curvature
