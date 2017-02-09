
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


class FFTOperator(LinearOperator):

    # ---Class attributes---

    default_codomain_dictionary = {RGSpace: RGSpace,
                                   HPSpace: LMSpace,
                                   GLSpace: LMSpace,
                                   LMSpace: HPSpace,
                                   }

    transformation_dictionary = {(RGSpace, RGSpace): RGRGTransformation,
                                 (HPSpace, LMSpace): HPLMTransformation,
                                 (GLSpace, LMSpace): GLLMTransformation,
                                 (LMSpace, HPSpace): LMHPTransformation,
                                 (LMSpace, GLSpace): LMGLTransformation
                                 }

    # ---Overwritten properties and methods---

    def __init__(self, domain=(), target=None, module=None):

        self._domain = self._parse_domain(domain)

        # Initialize domain and target
        if len(self.domain) != 1:
            raise ValueError(
                    'ERROR: TransformationOperator accepts only exactly one '
                    'space as input domain.')

        if target is None:
            target = (self.get_default_codomain(self.domain[0]), )
        self._target = self._parse_domain(target)

        # Create transformation instances
        try:
            forward_class = self.transformation_dictionary[
                (self.domain[0].__class__, self.target[0].__class__)]
        except KeyError:
            raise ValueError(
                "No forward transformation for domain-target pair "
                "found.")
        try:
            backward_class = self.transformation_dictionary[
                (self.target[0].__class__, self.domain[0].__class__)]
        except KeyError:
            raise ValueError(
                "No backward transformation for domain-target pair "
                "found.")

        self._forward_transformation = TransformationCache.create(
            forward_class, self.domain[0], self.target[0], module=module)

        self._backward_transformation = TransformationCache.create(
            backward_class, self.target[0], self.domain[0], module=module)

    def _times(self, x, spaces, dtype=None):
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

        result_field = x.copy_empty(domain=result_domain, dtype=dtype)
        result_field.set_val(new_val=new_val, copy=False)

        return result_field

    def _inverse_times(self, x, spaces, dtype=None):
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

        result_field = x.copy_empty(domain=result_domain, dtype=dtype)
        result_field.set_val(new_val=new_val, copy=False)

        return result_field

    # ---Mandatory properties and methods---

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def implemented(self):
        return True

    @property
    def unitary(self):
        return True

    # ---Added properties and methods---

    @classmethod
    def get_default_codomain(cls, domain):
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
