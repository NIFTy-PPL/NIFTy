from __future__ import absolute_import, division, print_function

from .compat import *
from .utilities import NiftyMetaBase


class Operator(NiftyMetaBase()):
    """Transforms values living on one domain into values living on another
    domain, and can also provide the Jacobian.
    """

    def __call__(self, x):
        """Returns transformed x

        Parameters
        ----------
        x : Linearization
            input

        Returns
        -------
        Linearization
            output
        """
        raise NotImplementedError
