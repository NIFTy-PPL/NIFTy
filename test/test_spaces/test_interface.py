import unittest
import numpy as np

from itertools import product
from d2o import distributed_data_object
from types import LambdaType
from numpy.testing import assert_, assert_raises, assert_equal
from nifty import RGSpace, LMSpace, GLSpace, HPSpace
from nifty.config import dependency_injector as di
from test.common import expand


def generate_spaces():
    spaces = [RGSpace(4)]

    if 'healpy' in di:
        spaces.append(HPSpace(4))
    if 'libsharp_wrapper_gl' in di:
        spaces.append(GLSpace(4))
    if 'healpy' in di or 'libsharp_wrapper_gl' in di:
        spaces.append(LMSpace(5))

    return spaces


class SpaceInterfaceTests(unittest.TestCase):
    def test_dependency_handling(self):
        if 'healpy' not in di and 'libsharp_wrapper_gl' not in di:
            with assert_raises(ImportError):
                LMSpace(5)
        elif 'healpy' not in di:
            with assert_raises(ImportError):
                HPSpace(4)
        elif 'libsharp_wrapper_gl' not in di:
            with assert_raises(ImportError):
                GLSpace(4)

    @expand(product(generate_spaces(), [['dtype', np.dtype],
                    ['harmonic', bool],
                    ['shape', tuple],
                    ['dim', int],
                    ['total_volume', np.float]]))
    def test_property_ret_type(self, space, attr_expected_type):
        assert_(
            isinstance(getattr(
                space,
                attr_expected_type[0]
            ), attr_expected_type[1])
        )

    @expand(product(generate_spaces(), [
        ['get_fft_smoothing_kernel_function', None, LambdaType],
        ['get_fft_smoothing_kernel_function', 2.0, LambdaType],
        ]))
    def test_method_ret_type(self, space, method_expected_type):

        assert_equal(
            type(getattr(
                space,
                method_expected_type[0])(*method_expected_type[1:-1])
            ),
            method_expected_type[-1]
        )
