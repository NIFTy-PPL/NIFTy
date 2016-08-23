from nifty.config import about
import nifty.nifty_utilities as utilities
from nifty.operators.linear_operator import LinearOperator
from transformations import TransformationFactory


class FFTOperator(LinearOperator):

    # ---Overwritten properties and methods---

    def __init__(self, domain=(), field_type=(), target=(),
                 field_type_target=(), implemented=True):
        super(FFTOperator, self).__init__(domain=domain,
                                          field_type=field_type,
                                          implemented=implemented)

        if self.domain == ():
            raise TypeError(about._errors.cstring(
                'ERROR: TransformationOperator needs a single space as '
                'input domain.'
            ))
        else:
            if len(self.domain) > 1:
                raise TypeError(about._errors.cstring(
                    'ERROR: TransformationOperator accepts only a single '
                    'space as input domain.'
                ))

        if self.field_type != ():
            raise TypeError(about._errors.cstring(
                'ERROR: TransformationOperator field-type has to be an '
                'empty tuple.'
            ))

        # currently not sanitizing the target
        self._target = self._parse_domain(
            utilities.get_default_codomain(self.domain[0])
        )
        self._field_type_target = self._parse_field_type(field_type_target)

        if self.field_type_target != ():
            raise TypeError(about._errors.cstring(
                'ERROR: TransformationOperator target field-type has to be an '
                'empty tuple.'
            ))

        self._forward_transformation = TransformationFactory.create(
            self.domain[0], self.target[0]
        )

        self._inverse_transformation = TransformationFactory.create(
            self.target[0], self.domain[0]
        )

    def adjoint_times(self, x, spaces=None, types=None):
        return self.inverse_times(x, spaces, types)

    def adjoint_inverse_times(self, x, spaces=None, types=None):
        return self.times(x, spaces, types)

    def inverse_adjoint_times(self, x, spaces=None, types=None):
        return self.times(x, spaces, types)

    def _times(self, x, spaces, types):
        spaces = utilities.cast_axis_to_tuple(spaces, len(x.domain))

        return self._forward_transformation.transform(x.val, axes=spaces)

    def _inverse_times(self, x, spaces, types):
        spaces = utilities.cast_axis_to_tuple(spaces, len(x.domain))

        return self._inverse_transformation.transform(x.val, axes=spaces)

    # ---Mandatory properties and methods---

    @property
    def target(self):
        return self._target

    @property
    def field_type_target(self):
        return self._field_type_target

