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
# Copyright(C) 2013-2023 Max-Planck-Society
# Authors: Philipp Frank, Vincent Eberle, Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np
from functools import reduce
from ..operators.endomorphic_operator import EndomorphicOperator, LinearOperator
from ..operators.scaling_operator import ScalingOperator
from ..domains.unstructured_domain import UnstructuredDomain
from ..domains.rg_space import RGSpace
from ..domains.irg_space import IRGSpace
from ..extra import is_fieldlike, is_operator
from ..sugar import makeOp, makeDomain, makeField


class _CumsumOperator(EndomorphicOperator):
    """
    Operator performs a cumulative sum along a space.

    Parameters:
    -----------
    domain: Domain or tuple of Domain or DomainTuple
        The domain on which the Operator's input Field is defined.
    space: None or int
        space for the integration

    Note: Integration domain must be of instance RGSpace or IRGSpace
    """
    def __init__(self, domain, space=None):
        self._domain = makeDomain(domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES

        if space is None:
            space = 0
        intdom = self._domain[space]

        if isinstance(intdom, RGSpace):
            if not len(intdom.distances) == 1:
                raise ValueError("Integration domain must be 1D")
            self._wgts = np.ones(intdom.shape)*intdom.distances[0]
        elif isinstance(intdom, IRGSpace):
            self._wgts = intdom.dvol
        else:
            raise ValueError("Integration domain of incorrect type!")

        self._wgts = np.sqrt(self._wgts)

        # spaces to axis
        self._axis = reduce(lambda a, b: a+b,
                            (len(dd.shape) for dd in self._domain[:space]), 0)
        _back = reduce(lambda a, b: a+b,
                       (len(dd.shape) for dd in self._domain[(space+1):]), 0)

        self._wgts = np.expand_dims(self._wgts,
                                    axis=tuple(i for i in range(self._axis)) +
                                    tuple(-(i+1) for i in range(_back)))

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = self._wgts*x.val
        if mode == self.ADJOINT_TIMES:
            x = np.flip(x, axis=self._axis)
        res = np.cumsum(x, axis=self._axis)
        if mode == self.ADJOINT_TIMES:
            res = np.flip(res, axis=self._axis)
        return makeField(self._domain, res)


def WPPrior(Amplitude, key='xi', space=None):
    """
    Models a Wiener Process or Brownian Motion, meaning that the differences
    along the considered space are Gaussian distirbutated. The Amplitude can be
    field or an operator and is scaling the Gaussian deviations before the
    cumulative Sum.

    Parameters:
    -----------
    Amplitude: :class:`nifty8.field.Field` or :class:`nifty.operators.Operator`
    key : String
        Key of Field containing the standard Gaussian distributed deviations.
    space:  None or int
        if None the Wiener Process is performed on the 0th space.
    """
    if is_fieldlike(Amplitude):
        wp = makeOp(Amplitude).ducktape(key)
    elif is_operator(Amplitude):
        wp = Amplitude * ScalingOperator(Amplitude.target, 1).ducktape(key)
    else:
        raise ValueError("Amplitude needs to be a field or an operator.")
    return _CumsumOperator(wp.target, space=space) @ wp


class IWPPrior(LinearOperator):
    """Operator that performs the integrations necessary for an integrated
    Wiener process.

    Parameters
    ----------
    iwp_domain: RGSpace or IRGSpace
        Domain for integrated Wiener Process.
    remaining_domain : DomainTuple, Domain or None
        All integrations are handled independently for this domain.
    """
    def __init__(self, iwp_domain, remaining_domain=None):
        # my_assert_isinstance(iwp_domain, IRGSpace)
        if remaining_domain is not None:
            self._target = makeDomain((iwp_domain, remaining_domain))
            dom = UnstructuredDomain((2, iwp_domain.size - 1)), remaining_domain
            self._domain = makeDomain(dom)
        else:
            self._target = makeDomain((iwp_domain))
            self._domain = makeDomain(UnstructuredDomain((2, iwp_domain.size - 1)))

        self._volumes = iwp_domain.distances
        if isinstance(iwp_domain, IRGSpace) and remaining_domain is not None:
            for _ in range(len(remaining_domain.shape)):
                self._volumes = self._volumes[..., np.newaxis]
        elif isinstance(iwp_domain, RGSpace):
            self._volumes = self._volumes[0]
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        first, second = (0,), (1,)
        from_second = (slice(1, None),)
        no_border = (slice(0, -1),)
        reverse = (slice(None, None, -1),)
        if mode == self.TIMES:
            x = x.val
            res = np.zeros(self._target.shape)
            res[from_second] = np.cumsum(x[second], axis=0)
            res[from_second] = (res[from_second] + res[no_border]) / 2 * self._volumes + x[first]
            res[from_second] = np.cumsum(res[from_second], axis=0)
        else:
            x = x.val_rw()
            res = np.zeros(self._domain.shape)
            x[from_second] = np.cumsum(x[from_second][reverse], axis=0)[reverse]
            res[first] += x[from_second]
            print(type(self._volumes))
            x[from_second] *= self._volumes / 2.0
            x[no_border] += x[from_second]
            res[second] += np.cumsum(x[from_second][reverse], axis=0)[reverse]
        return makeField(self._tgt(mode), res)
