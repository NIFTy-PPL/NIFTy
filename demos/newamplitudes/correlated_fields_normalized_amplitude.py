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
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import nifty5 as ift

if __name__ == '__main__':
    import numpy as np
    np.random.seed(42)
    nspecs = 2
    a = []
    spaces = []
    for _ in range(nspecs):
        ndim = 2
        sspace = ift.RGSpace(
            np.linspace(16, 20, num=ndim).astype(np.int),
            np.linspace(2.3, 7.99, num=ndim))
        hspace = sspace.get_default_codomain()
        spaces.append(sspace)
        target = ift.PowerSpace(hspace)
        a.append(ift.NormalizedAmplitude(target, 16, 1, 1, -3, 1, 0, 1, 0, 1))
    tgt = ift.makeDomain(tuple(spaces))
    op = ift.CorrelatedFieldNormAmplitude(tgt, a, 0., 1.)
    fld = ift.from_random('normal', op.domain)
    ift.extra.check_jacobian_consistency(op, fld)
