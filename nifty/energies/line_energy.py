# -*- coding: utf-8 -*-

from .energy import Energy


class LineEnergy(Energy):
    def __init__(self, position, energy, line_direction, zero_point=None):
        super(LineEnergy, self).__init__(position=position)
        self.line_direction = line_direction

        if zero_point is None:
            zero_point = energy.position
        self._zero_point = zero_point

        position_on_line = self._zero_point + self.position*line_direction
        self.energy = energy.at(position=position_on_line)

    def at(self, position):
        return self.__class__(position,
                              self.energy,
                              self.line_direction,
                              zero_point=self._zero_point)

    @property
    def value(self):
        return self.energy.value

    @property
    def gradient(self):
        return self.energy.gradient.dot(self.line_direction)

    @property
    def curvature(self):
        return self.energy.curvature
