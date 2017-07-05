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

import numpy as np

from ... import nifty_utilities as utilities
from ...spaces import RGSpace,\
                      GLSpace,\
                      HPSpace,\
                      LMSpace

from ..linear_operator import LinearOperator
from .transformations import RGRGTransformation,\
                            LMGLTransformation,\
                            LMHPTransformation,\
                            GLLMTransformation,\
                            HPLMTransformation,\
                            TransformationCache


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
        uniquely determined (except for "zerocenter").
        For GLSpace, HPSpace, and LMSpace, a sensible (but not unique)
        co-domain is chosen that should work satisfactorily in most situations,
        but for full control, the user should explicitly specify a codomain.
    module: String (optional)
        Software module employed for carrying out the transform operations.
        For RGSpace pairs this can be "scalar" or "mpi", where "scalar" is
        always available (using pyfftw if available, else numpy.fft), and "mpi"
        requires pyfftw and offers MPI parallelization.
        For sphere-related domains, only "pyHealpix" is
        available. If omitted, "fftw" is selected for RGSpaces if available,
        else "numpy"; on the sphere the default is "pyHealpix".
    domain_dtype: data type (optional)
        Data type of the fields that go into "times" and come out of
        "adjoint_times". Default is "numpy.complex".
    target_dtype: data type (optional)
        Data type of the fields that go into "adjoint_times" and come out of
        "times". Default is "numpy.complex".

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

    def __init__(self, domain, target=None, module=None,
                 domain_dtype=None, target_dtype=None, default_spaces=None):
        super(FFTOperator, self).__init__(default_spaces)

        # Initialize domain and target

        self._domain = self._parse_domain(domain)
        if len(self.domain) != 1:
            raise ValueError("TransformationOperator accepts only exactly one "
                             "space as input domain.")

        if target is None:
            target = (self.get_default_codomain(self.domain[0]), )
        self._target = self._parse_domain(target)
        if len(self.target) != 1:
            raise ValueError("TransformationOperator accepts only exactly one "
                             "space as output target.")

        # Create transformation instances
        forward_class = self.transformation_dictionary[
                (self.domain[0].__class__, self.target[0].__class__)]
        backward_class = self.transformation_dictionary[
                (self.target[0].__class__, self.domain[0].__class__)]

        self._forward_transformation = TransformationCache.create(
            forward_class, self.domain[0], self.target[0], module=module)

        self._backward_transformation = TransformationCache.create(
            backward_class, self.target[0], self.domain[0], module=module)

        # Store the dtype information
        if domain_dtype is None:
            self.logger.info("Setting domain_dtype to np.complex.")
            self.domain_dtype = np.complex
        else:
            self.domain_dtype = np.dtype(domain_dtype)

        if target_dtype is None:
            self.logger.info("Setting target_dtype to np.complex.")
            self.target_dtype = np.complex
        else:
            self.target_dtype = np.dtype(target_dtype)

    def _times(self, x, spaces):
        spaces = utilities.cast_axis_to_tuple(spaces, len(x.domain))
        if spaces is None:
            # this case means that x lives on only one space, which is
            # identical to the space in the domain of `self`. Otherwise the
            # input check of LinearOperator would have failed.
            axes = x.domain_axes[0]
        else:
            axes = x.domain_axes[spaces[0]]

        new_val = self._forward_transformation.transform(x.val, axes=axes)

        if spaces is None:
            result_domain = self.target
        else:
            result_domain = list(x.domain)
            result_domain[spaces[0]] = self.target[0]

        result_field = x.copy_empty(domain=result_domain,
                                    dtype=self.target_dtype)
        result_field.set_val(new_val=new_val, copy=True)

        return result_field

    def _adjoint_times(self, x, spaces):
        spaces = utilities.cast_axis_to_tuple(spaces, len(x.domain))
        if spaces is None:
            # this case means that x lives on only one space, which is
            # identical to the space in the domain of `self`. Otherwise the
            # input check of LinearOperator would have failed.
            axes = x.domain_axes[0]
        else:
            axes = x.domain_axes[spaces[0]]

        new_val = self._backward_transformation.transform(x.val, axes=axes)

        if spaces is None:
            result_domain = self.domain
        else:
            result_domain = list(x.domain)
            result_domain[spaces[0]] = self.domain[0]

        result_field = x.copy_empty(domain=result_domain,
                                    dtype=self.domain_dtype)
        result_field.set_val(new_val=new_val, copy=True)

        return result_field

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

    # ---Added properties and methods---

    @classmethod
    def get_default_codomain(cls, domain):
        """Returns a codomain to the given domain.

        Parameters
        ----------
        domain: Space
            An instance of RGSpace, HPSpace, GLSpace or LMSpace.

        Returns
        -------
        target: Space
            A (more or less perfect) counterpart to "domain" with respect
            to a FFT operation.
            Whenever "domain" is an RGSpace, the codomain (and its parameters)
            are uniquely determined (except for "zerocenter").
            For GLSpace, HPSpace, and LMSpace, a sensible (but not unique)
            co-domain is chosen that should work satisfactorily in most
            situations. For full control however, the user should not rely on
            this method.

        Raises
        ------
        ValueError:
            if no default codomain is defined for "domain".

        """
        domain_class = domain.__class__
        try:
            codomain_class = cls.default_codomain_dictionary[domain_class]
        except KeyError:
            raise ValueError("Unknown domain")

        try:
            transform_class = cls.transformation_dictionary[(domain_class,
                                                             codomain_class)]
        except KeyError:
            raise ValueError(
                "No transformation for domain-codomain pair found.")

        return transform_class.get_codomain(domain)
