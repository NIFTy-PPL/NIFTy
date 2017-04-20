# NIFTy
# Copyright (C) 2017  Theo Steininger
#
# Author: Theo Steininger
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import unittest

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
