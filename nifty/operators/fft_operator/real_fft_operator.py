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

import nifty.nifty_utilities as utilities
from nifty.spaces import RGSpace,\
                         GLSpace,\
                         HPSpace,\
                         LMSpace

from nifty.operators.linear_operator import LinearOperator
from transformations import RGRGTransformation,\
                            LMGLTransformation,\
                            LMHPTransformation,\
                            GLLMTransformation,\
                            HPLMTransformation,\
                            TransformationCache


class RealFFTOperator(LinearOperator):
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

    In contrast to the FFTOperator, RealFFTOperator accepts and returns
    real-valued fields only. For the harmonic-space counterpart of a
    real-valued field living on an RGSpace, the sum of real
    and imaginary components is stored. Since the full complex field has
    Hermitian symmetry, this is sufficient to reconstruct the full field
    whenever needed (e.g. during the transform back to position space).

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
    module: String (optional)
        Software module employed for carrying out the transform operations.
        For RGSpace pairs this can be "scalar" or "mpi", where "scalar" is
        always available (using pyfftw if available, else numpy.fft), and "mpi"
        requires pyfftw and offers MPI parallelization.
        For sphere-related domains, only "pyHealpix" is
        available. If omitted, "fftw" is selected for RGSpaces if available,
        else "numpy"; on the sphere the default is "pyHealpix".

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

    def __init__(self, domain, target=None, module=None,
                 default_spaces=None):
        super(RealFFTOperator, self).__init__(default_spaces)

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

    def _prep(self, x, spaces, dom):
        assert issubclass(x.dtype.type,np.floating), \
            "Argument must be real-valued"
        spaces = utilities.cast_axis_to_tuple(spaces, len(x.domain))
        if spaces is None:
            # this case means that x lives on only one space, which is
            # identical to the space in the domain of `self`. Otherwise the
            # input check of LinearOperator would have failed.
            axes = x.domain_axes[0]
        else:
            axes = x.domain_axes[spaces[0]]

        if spaces is None:
            result_domain = dom
        else:
            result_domain = list(x.domain)
            result_domain[spaces[0]] = dom[0]

        result_field = x.copy_empty(domain=result_domain, dtype=x.dtype)
        return spaces, axes, result_field

    def _times(self, x, spaces):
        spaces, axes, result_field = self._prep(x, spaces, self.target)

        if type(self._domain[0]) != RGSpace:
            new_val = self._forward_transformation.transform(x.val, axes=axes)
            result_field.set_val(new_val=new_val, copy=True)
            return result_field

        if self._target[0].harmonic:  # going to harmonic space
            new_val = self._forward_transformation.transform(x.val, axes=axes)
            result_field.set_val(new_val=new_val.real+new_val.imag)
        else:
            tval = self._domain[0].hermitianize_inverter(x.val, axes)
            tval = 0.5*((x.val+tval)+1j*(x.val-tval))
            new_val = self._forward_transformation.transform(tval, axes=axes)
            result_field.set_val(new_val=new_val.real)
        return result_field

    def _adjoint_times(self, x, spaces):
        spaces, axes, result_field = self._prep(x, spaces, self.domain)

        if type(self._domain[0]) != RGSpace:
            new_val = self._backward_transformation.transform(x.val, axes=axes)
            result_field.set_val(new_val=new_val, copy=True)
            return result_field

        if self._domain[0].harmonic:  # going to harmonic space
            new_val = self._backward_transformation.transform(x.val, axes=axes)
            result_field.set_val(new_val=new_val.real+new_val.imag)
        else:
            tval = self._target[0].hermitianize_inverter(x.val, axes)
            tval = 0.5*((x.val+tval)+1j*(x.val-tval))
            new_val = self._backward_transformation.transform(tval, axes=axes)
            result_field.set_val(new_val=new_val.real)
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
            are uniquely determined.
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
