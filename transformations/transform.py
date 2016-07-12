from nifty import RGSpace
from nifty.config import about

import numpy as np


class Transform(object):
    """
        A generic fft object without any implementation.
    """


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
