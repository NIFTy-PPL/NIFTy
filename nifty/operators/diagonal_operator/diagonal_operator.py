# -*- coding: utf-8 -*-

import numpy as np

from d2o import distributed_data_object,\
                STRATEGIES as DISTRIBUTION_STRATEGIES

from nifty.config import about,\
                         nifty_configuration as gc
from nifty.field import Field
from nifty.operators.endomorphic_operator import EndomorphicOperator


class DiagonalOperator(EndomorphicOperator):

    # ---Overwritten properties and methods---

    def __init__(self, domain=(), field_type=(), implemented=True,
                 diagonal=None, bare=False, copy=True,
                 distribution_strategy=None):
        self._domain = self._parse_domain(domain)
        self._field_type = self._parse_field_type(field_type)

        self._implemented = bool(implemented)

        if distribution_strategy is None:
            if isinstance(diagonal, distributed_data_object):
                distribution_strategy = diagonal.distribution_strategy
            elif isinstance(diagonal, Field):
                distribution_strategy = diagonal.distribution_strategy

        self._distribution_strategy = self._parse_distribution_strategy(
                               distribution_strategy=distribution_strategy,
                               val=diagonal)

        self.set_diagonal(diagonal=diagonal, bare=bare, copy=copy)

    def _times(self, x, spaces, types):
        return self._times_helper(x, spaces, types,
                                  operation=lambda z: z.__mul__)

    def _adjoint_times(self, x, spaces, types):
        return self._times_helper(x, spaces, types,
                                  operation=lambda z: z.adjoint().__mul__)

    def _inverse_times(self, x, spaces, types):
        return self._times_helper(x, spaces, types,
                                  operation=lambda z: z.__rdiv__)

    def _adjoint_inverse_times(self, x, spaces, types):
        return self._times_helper(x, spaces, types,
                                  operation=lambda z: z.adjoint().__rdiv__)

    def diagonal(self, bare=False, copy=True):
        if bare:
            diagonal = self._diagonal.weight(power=-1)
        elif copy:
            diagonal = self._diagonal.copy()
        else:
            diagonal = self._diagonal
        return diagonal

    def inverse_diagonal(self, bare=False):
        return 1/self.diagonal(bare=bare, copy=False)

    def trace(self, bare=False):
        return self.diagonal(bare=bare, copy=False).sum()

    def inverse_trace(self, bare=False):
        return self.inverse_diagonal(bare=bare, copy=False).sum()

    def trace_log(self):
        log_diagonal = self.diagonal(copy=False).apply_scalar_function(np.log)
        return log_diagonal.sum()

    def determinant(self):
        return self.diagonal(copy=False).val.prod()

    def inverse_determinant(self):
        return 1/self.determinant()

    def log_determinant(self):
        return np.log(self.determinant())

    # ---Mandatory properties and methods---

    @property
    def domain(self):
        return self._domain

    @property
    def field_type(self):
        return self._field_type

    @property
    def implemented(self):
        return self._implemented

    @property
    def symmetric(self):
        return self._symmetric

    @property
    def unitary(self):
        return self._unitary

    # ---Added properties and methods---

    @property
    def distribution_strategy(self):
        return self._distribution_strategy

    def _parse_distribution_strategy(self, distribution_strategy, val):
        if distribution_strategy is None:
            if isinstance(val, distributed_data_object):
                distribution_strategy = val.distribution_strategy
            elif isinstance(val, Field):
                distribution_strategy = val.distribution_strategy
            else:
                about.warnings.cprint("WARNING: Datamodel set to default!")
                distribution_strategy = gc['default_distribution_strategy']
        elif distribution_strategy not in DISTRIBUTION_STRATEGIES['all']:
            raise ValueError(about._errors.cstring(
                    "ERROR: Invalid distribution_strategy!"))
        return distribution_strategy

    def set_diagonal(self, diagonal, bare=False, copy=True):
        # use the casting functionality from Field to process `diagonal`
        f = Field(domain=self.domain,
                  val=diagonal,
                  field_type=self.field_type,
                  distribution_strategy=self.distribution_strategy,
                  copy=copy)

        # weight if the given values were `bare` and `implemented` is True
        # do inverse weightening if the other way around
        if bare and self.implemented:
            # If `copy` is True, we won't change external data by weightening
            # Otherwise, inplace weightening would change the external field
            f.weight(inplace=copy)
        elif not bare and not self.implemented:
            # If `copy` is True, we won't change external data by weightening
            # Otherwise, inplace weightening would change the external field
            f.weight(inplace=copy, power=-1)

        # check if the operator is symmetric:
        self._symmetric = (f.val.imag == 0).all()

        # check if the operator is unitary:
        self._unitary = (f.val * f.val.conjugate() == 1).all()

        # store the diagonal-field
        self._diagonal = f

    def _times_helper(self, x, spaces, types, operation):
        # if the domain and field_type match directly
        # -> multiply the fields directly
        if x.domain == self.domain and x.field_type == self.field_type:
            # here the actual multiplication takes place
            return operation(self.diagonal(copy=False))(x)

        # if the distribution_strategy of self is sub-slice compatible to
        # the one of x, reshape the local data of self and apply it directly
        active_axes = []
        if spaces is None:
            if self.domain != ():
                for axes in x.domain_axes:
                    active_axes += axes
        else:
            for space_index in spaces:
                active_axes += x.domain_axes[space_index]

        if types is None:
            if self.field_type != ():
                for axes in x.field_type_axes:
                    active_axes += axes
        else:
            for type_index in types:
                active_axes += x.field_type_axes[type_index]

        axes_local_distribution_strategy = \
            x.val.get_axes_local_distribution_strategy(active_axes)
        if axes_local_distribution_strategy == self.distribution_strategy:
            local_diagonal = self._diagonal.val.get_local_data(copy=False)
        else:
            # create an array that is sub-slice compatible
            redistr_diagonal_val = self._diagonal.val.copy(
                distribution_strategy=axes_local_distribution_strategy)
            local_diagonal = redistr_diagonal_val.get_local_data(copy=False)

        reshaper = [x.shape[i] if i in active_axes else 1
                    for i in xrange(len(x.shape))]
        reshaped_local_diagonal = np.reshape(local_diagonal, reshaper)

        # here the actual multiplication takes place
        local_result = operation(reshaped_local_diagonal)(
                           x.val.get_local_data(copy=False))

        result_field = x.copy_empty(dtype=local_result.dtype)
        result_field.val.set_local_data(local_result, copy=False)
        return result_field
