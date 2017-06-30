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
#
# Copyright(C) 2013-2017 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

import unittest

from numpy.testing import assert_equal
from keepers import Repository
from test.common import expand, generate_spaces
from nose.plugins.skip import SkipTest
import os

class SpaceSerializationTests(unittest.TestCase):
    @expand([[space] for space in generate_spaces()])
    def test_serialization(self, space):
        try:
            import h5py
        except ImportError:
            raise SkipTest
        try:
            os.remove('test.h5')
        except:
            pass
        repo = Repository('test.h5')
        repo.add(space, 'space')
        repo.commit()
        assert_equal(space, repo.get('space'))
        os.remove('test.h5')
