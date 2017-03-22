import unittest
import numpy as np

from itertools import product
from types import LambdaType
from numpy.testing import assert_, assert_raises, assert_equal
from nose.plugins.skip import SkipTest
from nifty import LMSpace, GLSpace, HPSpace
from nifty.config import dependency_injector as di
from test.common import expand, generate_spaces


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
        # handle exceptions here
        try:
            getattr(
                space, method_expected_type[0])(*method_expected_type[1:-1])
        except NotImplementedError:
            raise SkipTest

        assert_equal(
            type(getattr(
                space,
                method_expected_type[0])(*method_expected_type[1:-1])
            ),
            method_expected_type[-1]
        )

    @expand([[space] for space in generate_spaces()])
    def test_copy(self, space):
        # make sure it's a deep copy
        assert_(space is not space.copy())
        # make sure contents are the same
        assert_equal(space, space.copy())
