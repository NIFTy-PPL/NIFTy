# NIFTy
# Copyright (C) 2017  Theo Steininger
#
# Author: Theo Steininger
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


class DiagonalProberMixin(object):
    def __init__(self, *args, **kwargs):
        self.reset()
        super(DiagonalProberMixin, self).__init__(*args, **kwargs)

    def reset(self):
        self.__sum_of_probings = 0
        self.__sum_of_squares = 0
        self.__diagonal = None
        self.__diagonal_variance = None
        super(DiagonalProberMixin, self).reset()

    def finish_probe(self, probe, pre_result):
        result = probe[1].conjugate()*pre_result
        self.__sum_of_probings += result
        if self.compute_variance:
            self.__sum_of_squares += result.conjugate() * result
        super(DiagonalProberMixin, self).finish_probe(probe, pre_result)

    @property
    def diagonal(self):
        if self.__diagonal is None:
            self.__diagonal = self.__sum_of_probings/self.probe_count
        return self.__diagonal

    @property
    def diagonal_variance(self):
        if not self.compute_variance:
            raise AttributeError("self.compute_variance is set to False")
        if self.__diagonal_variance is None:
            # variance = 1/(n-1) (sum(x^2) - 1/n*sum(x)^2)
            n = self.probe_count
            sum_pr = self.__sum_of_probings
            mean = self.diagonal
            sum_sq = self.__sum_of_squares

            self.__diagonal_variance = ((sum_sq - sum_pr*mean) / (n-1))
        return self.__diagonal_variance
