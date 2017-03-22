import unittest

from nifty import dependency_injector as di

from numpy.testing import assert_equal
from keepers import Repository
from test.common import expand, generate_spaces

try:
    import h5py
except ImportError:
    h5py_available = False
else:
    h5py_available = True

if h5py_available:
    class SpaceSerializationTests(unittest.TestCase):
        # variable to hold the repository
        _repo = None

        @classmethod
        def setUpClass(cls):
            cls._repo = Repository('test.h5')

        @expand([[space] for space in generate_spaces()])
        def test_serialization(self, space):
            self._repo.add(space, 'space')
            self._repo.commit()

            assert_equal(space, self._repo.get('space'))
