from nifty import RGSpace
from nifty.config import about

import numpy as np


class Transform(object):
    """
        A generic fft object without any implementation.
    """

    @staticmethod
    def check_codomain(domain, codomain):
        if codomain is None:
            return False

        if isinstance(domain, RGSpace):
            if not isinstance(codomain, RGSpace):
                raise TypeError(about._errors.cstring(
                    "ERROR: The given codomain must be a rg_space."
                ))

            if not np.all(np.array(domain.paradict['shape']) ==
                                  np.array(codomain.paradict['shape'])):
                return False

            if domain.harmonic == codomain.harmonic:
                return False

            # Check complexity
            # Prepare shorthands
            dcomp = domain.paradict['complexity']
            cocomp = codomain.paradict['complexity']

            # Case 1: if domain is completely complex, the codomain
            # must be complex too
            if dcomp == 2:
                if cocomp != 2:
                    return False
            # Case 2: if domain is hermitian, the codomain can be
            # real, a warning is raised otherwise
            elif dcomp == 1:
                if cocomp > 0:
                    about.warnings.cprint(
                        "WARNING: Unrecommended codomain! " +
                        "The domain is hermitian, hence the" +
                        "codomain should be restricted to real values."
                    )
            # Case 3: if domain is real, the codomain should be hermitian
            elif dcomp == 0:
                if cocomp == 2:
                    about.warnings.cprint(
                        "WARNING: Unrecommended codomain! " +
                        "The domain is real, hence the" +
                        "codomain should be restricted to" +
                        "hermitian configuration."
                    )
                elif cocomp == 0:
                    return False

            # Check if the distances match, i.e. dist' = 1 / (num * dist)
            if not np.all(
                np.absolute(np.array(domain.paradict['shape']) *
                            np.array(domain.distances) *
                            np.array(codomain.distances) - 1) < domain.epsilon):
                return False
        else:
            return False

        return True

    def __init__(self, domain, codomain):
        pass

    def transform(self, val, axes, **kwargs):
        """
            A generic ff-transform function.

            Parameters
            ----------
            field_val : distributed_data_object
                The value-array of the field which is supposed to
                be transformed.

            domain : nifty.rg.nifty_rg.rg_space
                The domain of the space which should be transformed.

            codomain : nifty.rg.nifty_rg.rg_space
                The taget into which the field should be transformed.
        """
        raise NotImplementedError
