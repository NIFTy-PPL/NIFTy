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
# Copyright(C) 2013-2019 Max-Planck-Society
# Authors: Philipp Frank
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from ..multi_domain import MultiDomain
from ..multi_field import MultiField
from .endomorphic_operator import EndomorphicOperator


class PartialConjugate(EndomorphicOperator):
    """Perform partial conjugation of a :class:`MultiField`

    Parameters
    ----------
    domain : MultiDomain
        The operator's input domain and output target
    conjugation_keys : iterable of string
        The keys of the :class:`MultiField` for which complex conjugation
        should be performed.
    """
    def __init__(self, domain, conjugation_keys):
        if not isinstance(domain, MultiDomain):
            raise ValueError("MultiDomain expected!")
        indom = (key in domain.keys() for key in conjugation_keys)
        if sum(indom) != len(conjugation_keys):
            raise ValueError("conjugation_keys not in domain!")
        self._domain = domain
        self._conjugation_keys = conjugation_keys
        self._capabilities = self._all_ops

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.to_dict()
        for k in self._conjugation_keys:
            x[k] = x[k].conjugate()
        return MultiField.from_dict(x, self._domain)
