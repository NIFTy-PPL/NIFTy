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

    def __init__(self, domain=(), field_type=(), implemented=False,
                 diagonal=None, bare=False, copy=True, datamodel=None):
        super(DiagonalOperator, self).__init__(domain=domain,
                                               field_type=field_type,
                                               implemented=implemented)

        self._implemented = bool(implemented)

        if datamodel is None:
            if isinstance(diagonal, distributed_data_object):
                datamodel = diagonal.distribution_strategy
            elif isinstance(diagonal, Field):
                datamodel = diagonal.datamodel

        self.datamodel = self._parse_datamodel(datamodel=datamodel,
                                               val=diagonal)

        self.set_diagonal(diagonal=diagonal, bare=bare, copy=copy)

    def _times(self, x, spaces, types):
        pass

    def _adjoint_times(self, x, spaces, types):
        pass

    def _inverse_times(self, x, spaces, types):
        pass

    def _adjoint_inverse_times(self, x, spaces, types):
        pass

    def _inverse_adjoint_times(self, x, spaces, types):
        pass

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
    def datamodel(self):
        return self._datamodel

    def _parse_datamodel(self, datamodel, val):
        if datamodel is None:
            if isinstance(val, distributed_data_object):
                datamodel = val.distribution_strategy
            elif isinstance(val, Field):
                datamodel = val.datamodel
            else:
                about.warnings.cprint("WARNING: Datamodel set to default!")
                datamodel = gc['default_datamodel']
        elif datamodel not in DISTRIBUTION_STRATEGIES['all']:
            raise ValueError(about._errors.cstring(
                    "ERROR: Invalid datamodel!"))
        return datamodel

    def set_diagonal(self, diagonal, bare=False, copy=True):
        # use the casting functionality from Field to process `diagonal`
        f = Field(domain=self.domain,
                  val=diagonal,
                  field_type=self.field_type,
                  datamodel=self.datamodel,
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
