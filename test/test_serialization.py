import unittest

from numpy.testing import assert_equal
from keepers import Repository
from test.common import expand, generate_spaces


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
