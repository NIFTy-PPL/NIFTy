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

from ..operators import LinearOperator


class SelectionOperator(LinearOperator):
    def __init__(self, domain, key):
        from ..multi import MultiDomain
        if not isinstance(domain, MultiDomain):
            raise TypeError("Domain must be a MultiDomain")
        self._target = domain[key]
        self._domain = domain
        self._key = key

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        # FIXME Is the copying necessary?
        self._check_input(x, mode)
        if mode == self.TIMES:
            return x[self._key].copy()
        else:
            from ..multi import MultiField
            return MultiField({self._key: x.copy()})
