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

import os

import keepers

# pre-create the D2O configuration instance and set its path explicitly
d2o_configuration = keepers.get_Configuration(
                    name='D2O',
                    file_name='D2O.conf',
                    search_paths=[os.path.expanduser('~') + "/.config/nifty/",
                                  os.path.expanduser('~') + "/.config/",
                                  './'])
