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
# Copyright(C) 2013-2017 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from ... import Field, nifty_utilities as utilities
from ...spaces import RGSpace,\
                      GLSpace,\
                      HPSpace,\
                      LMSpace

from ..linear_operator import LinearOperator
from .transformations import RGRGTransformation,\
                            LMGLTransformation,\
                            LMHPTransformation,\
                            GLLMTransformation,\
                            HPLMTransformation


class FFTOperator(LinearOperator):
    """Transforms between a pair of position and harmonic domains.

    Built-in domain pairs are
      - a harmonic and a non-harmonic RGSpace (with matching distances)
      - a HPSpace and a LMSpace
      - a GLSpace and a LMSpace
    Within a domain pair, both orderings are possible.

    The operator provides a "times" and an "adjoint_times" operation.
    For a pair of RGSpaces, the "adjoint_times" operation is equivalent to
    "inverse_times"; for the sphere-related domains this is not the case, since
    the operator matrix is not square.

    Parameters
    ----------
    domain: Space or single-element tuple of Spaces
        The domain of the data that is input by "times" and output by
        "adjoint_times".
    target: Space or single-element tuple of Spaces (optional)
        The domain of the data that is output by "times" and input by
        "adjoint_times".
        If omitted, a co-domain will be chosen automatically.
        Whenever "domain" is an RGSpace, the codomain (and its parameters) are
        uniquely determined.
        For GLSpace, HPSpace, and LMSpace, a sensible (but not unique)
        co-domain is chosen that should work satisfactorily in most situations,
        but for full control, the user should explicitly specify a codomain.

    Attributes
    ----------
    domain: Tuple of Spaces (with one entry)
        The domain of the data that is input by "times" and output by
        "adjoint_times".
    target: Tuple of Spaces (with one entry)
        The domain of the data that is output by "times" and input by
        "adjoint_times".
    unitary: bool
        Returns True if the operator is unitary (currently only the case if
        the domain and codomain are RGSpaces), else False.

    Raises
    ------
    ValueError:
        if "domain" or "target" are not of the proper type.

    """

    # ---Class attributes---

    default_codomain_dictionary = {RGSpace: RGSpace,
                                   HPSpace: LMSpace,
                                   GLSpace: LMSpace,
                                   LMSpace: GLSpace,
                                   }

    transformation_dictionary = {(RGSpace, RGSpace): RGRGTransformation,
                                 (HPSpace, LMSpace): HPLMTransformation,
                                 (GLSpace, LMSpace): GLLMTransformation,
                                 (LMSpace, HPSpace): LMHPTransformation,
                                 (LMSpace, GLSpace): LMGLTransformation
                                 }

    # ---Overwritten properties and methods---

    def __init__(self, domain, target=None, default_spaces=None):
        super(FFTOperator, self).__init__(default_spaces)

        # Initialize domain and target
        self._domain = self._parse_domain(domain)
        if len(self.domain) != 1:
            raise ValueError("TransformationOperator accepts only exactly one "
                             "space as input domain.")

        if target is None:
            target = (self.domain[0].get_default_codomain(), )
        self._target = self._parse_domain(target)
        if len(self.target) != 1:
            raise ValueError("TransformationOperator accepts only exactly one "
                             "space as output target.")
        self.domain[0].check_codomain(self.target[0])
        self.target[0].check_codomain(self.domain[0])

        # Create transformation instances
        forward_class = self.transformation_dictionary[
                (self.domain[0].__class__, self.target[0].__class__)]
        backward_class = self.transformation_dictionary[
                (self.target[0].__class__, self.domain[0].__class__)]

        self._forward_transformation = forward_class(
            self.domain[0], self.target[0])

        self._backward_transformation = backward_class(
            self.target[0], self.domain[0])

    def _times(self, x, spaces):
        spaces = utilities.cast_axis_to_tuple(spaces, len(x.domain))
        if spaces is None:
            # this case means that x lives on only one space, which is
            # identical to the space in the domain of `self`. Otherwise the
            # input check of LinearOperator would have failed.
            axes = x.domain_axes[0]
            result_domain = self.target
        else:
            axes = x.domain_axes[spaces[0]]
            result_domain = list(x.domain)
            result_domain[spaces[0]] = self.target[0]

        new_val = self._forward_transformation.transform(x.val, axes=axes)
        return Field(result_domain, new_val, copy=False)

    def _adjoint_times(self, x, spaces):
        spaces = utilities.cast_axis_to_tuple(spaces, len(x.domain))
        if spaces is None:
            # this case means that x lives on only one space, which is
            # identical to the space in the domain of `self`. Otherwise the
            # input check of LinearOperator would have failed.
            axes = x.domain_axes[0]
            result_domain = self.domain
        else:
            axes = x.domain_axes[spaces[0]]
            result_domain = list(x.domain)
            result_domain[spaces[0]] = self.domain[0]

        new_val = self._backward_transformation.transform(x.val, axes=axes)
        return Field(result_domain, new_val, copy=False)

    # ---Mandatory properties and methods---

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def unitary(self):
        return (self._forward_transformation.unitary and
                self._backward_transformation.unitary)
