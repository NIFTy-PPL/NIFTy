from nifty.config import about
import nifty.nifty_utilities as utilities
from nifty.operators.linear_operator import LinearOperator
from transformations import TransformationFactory


class FFTOperator(LinearOperator):

    # ---Overwritten properties and methods---

    def __init__(self, domain=(), field_type=(), target=None):
        super(FFTOperator, self).__init__(domain=domain,
                                          field_type=field_type)

        if len(self.domain) != 1:
            raise ValueError(about._errors.cstring(
                    'ERROR: TransformationOperator accepts only exactly one '
                    'space as input domain.'))

        if self.field_type != ():
            raise ValueError(about._errors.cstring(
                'ERROR: TransformationOperator field-type must be an '
                'empty tuple.'
            ))

        if target is None:
            target = utilities.get_default_codomain(self.domain[0])

        self._target = self._parse_domain(
                        utilities.get_default_codomain(self.domain[0]))

        self._forward_transformation = TransformationFactory.create(
            self.domain[0], self.target[0]
        )

        self._inverse_transformation = TransformationFactory.create(
            self.target[0], self.domain[0]
        )

    def _times(self, x, spaces, types):
        spaces = utilities.cast_axis_to_tuple(spaces, len(x.domain))

        axes = x.domain_axes[spaces[0]]
        new_val = self._forward_transformation.transform(x.val, axes=axes)

        if spaces is None:
            result_domain = self.target
        else:
            result_domain = list(x.domain)
            result_domain[spaces[0]] = self.target[0]

        result_field = x.copy_empty(domain=result_domain)
        result_field.set_val(new_val=new_val)

        return result_field

    def _inverse_times(self, x, spaces, types):
        spaces = utilities.cast_axis_to_tuple(spaces, len(x.domain))

        axes = x.domain_axes[spaces[0]]
        new_val = self._inverse_transformation.transform(x.val, axes=axes)

        if spaces is None:
            result_domain = self.domain
        else:
            result_domain = list(x.domain)
            result_domain[spaces[0]] = self.domain[0]

        result_field = x.copy_empty(domain=result_domain)
        result_field.set_val(new_val=new_val)

        return result_field

    # ---Mandatory properties and methods---

    @property
    def target(self):
        return self._target

    @property
    def field_type_target(self):
        return self.field_type

    @property
    def implemented(self):
        return True

    @property
    def unitary(self):
        return True
