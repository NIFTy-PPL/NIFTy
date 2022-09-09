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
# Copyright(C) 2022 Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import sys

sys.path.append("demos")

import tempfile

import nifty8 as ift
from getting_started_7_config_file import builder_dct


def test_optimize_kl_cfg_save_and_load():
    cfg = ift.OptimizeKLConfig.from_file("demos/getting_started_7_config_file.cfg", builder_dct)
    with tempfile.NamedTemporaryFile() as f:
        file_name = f.name
        cfg.to_file(file_name)
        cfg0 = ift.OptimizeKLConfig.from_file(file_name, builder_dct)
        assert cfg == cfg0
    # /Move to tests
