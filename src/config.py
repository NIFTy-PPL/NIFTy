# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
#
# Copyright 2024 Gordian Edenhofer
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
# Copyright(C) 2024 Max-Planck-Society


_config = dict(
    hartley_convention="non_canonical_hartley",
    break_on_device_copy=False,
    fail_on_device_copy=False,
    fail_on_nontrivial_anyarray_creation_on_host=False,
)


def update(key, value, /):
    """Update the global configuration of NIFTy and NIFTy.re

    Parameters
    ----------
    key : str
        Identifier for the configuration option.
    value : Any
        Value for the configuration option.


    Currently, the following configuration options are available:

    - "hartley_convention": one of "non_canonical_hartley" or
      "canonical_hartley" for ducc's old non-canonical Hartley convention
      respectively ducc's new canononical Hartley convention
    """
    global _config
    if not isinstance(key, str):
        raise TypeError(f"key must be a string; got {key!r}")
    key = key.lower()
    if key == "hartley_convention":
        if not isinstance(value, str):
            raise TypeError(f"value to {key!r} must be a string; got {value!r}")
        if value in ("ducc_hartley", "non_canonical_hartley"):
            value = "non_canonical_hartley"
        elif value in ("ducc_fht", "canonical_hartley"):
            value = "canonical_hartley"
        else:
            raise ValueError(f"invalid value to {key!r}; got {value!r}")
    _config[key] = value
