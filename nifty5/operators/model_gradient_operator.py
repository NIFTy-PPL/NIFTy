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
#
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from ..multi.multi_domain import MultiDomain
from ..multi.multi_field import MultiField
from ..sugar import full
from .linear_operator import LinearOperator


class ModelGradientOperator(LinearOperator):
    def __init__(self, gradients, domain, target):
        super(ModelGradientOperator, self).__init__()
        self._gradients = gradients
        gradients_domain = MultiField(self._gradients).domain
        self._domain = MultiDomain.make(domain)

        # Check compatibility
        if not (set(gradients_domain.items()) <= set(self.domain.items())):
            raise ValueError

        self._target = target
        for grad in gradients.values():
            if self._target != grad.target:
                raise TypeError(
                    'All gradients have to have the same target domain')

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES

    @property
    def gradients(self):
        return self._gradients

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            res = None
            for key, op in self._gradients.items():
                if res is None:
                    res = op(x[key])
                else:
                    res += op(x[key])
            # Needed if gradients == {}
            if res is None:
                res = full(self.target, 0.)
            if not res.domain == self.target:
                raise TypeError
        else:
            grad_keys = self._gradients.keys()
            res = {}
            for dd in self.domain:
                if dd in grad_keys:
                    res[dd] = self._gradients[dd].adjoint_times(x)
                # else:
                    # res[dd] = full(self.domain[dd], 0.)
            res = MultiField(res)
            # if not res.domain == self.domain:
            #     raise TypeError
        return res
