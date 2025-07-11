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
# Copyright(C) 2025 Philipp Arras

from tempfile import TemporaryDirectory


class MPISafeTempdir:
    def __init__(self, comm):
        self._comm = comm
        if self._master:
            self._direc = TemporaryDirectory()
            self._name = self._direc.name
        else:
            self._name = None
        if self._comm is not None:
            self._name = self._comm.bcast(self._name, root=0)
        assert self._name is not None

    def _barrier(self):
        if self._comm is not None:
            self._comm.Barrier()

    @property
    def _master(self):
        return self._comm is None or self._comm.Get_rank() == 0

    def __enter__(self):
        self._barrier()
        if self._master:
            self._direc.__enter__()
        return self._name

    def __exit__(self, *args, **kwargs):
        self._barrier()
        if self._master:
            self._direc.__exit__(*args, **kwargs)
