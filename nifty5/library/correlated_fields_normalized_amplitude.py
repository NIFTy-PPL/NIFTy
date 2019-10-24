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
from functools import reduce
from operator import mul

def CorrelatedFieldNormAmplitude(target, amplitudes,
                                 stdmean,
                                 stdstd,
                                 names=['xi','std']):
    """Constructs an operator which turns white Gaussian excitation fields
    into a correlated field defined on a DomainTuple with n entries and n
    separate correlation structures.

    This operator may be used as a model for multi-frequency reconstructions
    with a correlation structure in both spatial and energy direction.

    Parameters
    ----------
    target : Domain, DomainTuple or tuple of Domain
        Target of the operator. Must contain exactly n spaces.
    amplitudes: Opertor, iterable of Operator
        Amplitude operator if n = 1 or list of n amplitude operators.
    stdmean : float
        Prior mean of the overall standart deviation.
    stdstd : float
        Prior standart deviation of the overall standart deviation.
    names : iterable of string
        :class:`MultiField` keys for xi-field and std-field.

    Returns
    -------
    Operator
        Correlated field

    Notes
    -----
    In NIFTy, non-harmonic RGSpaces are by definition periodic. Therefore
    the operator constructed by this method will output a correlated field
    with *periodic* boundary conditions. If a non-periodic field is needed,
    one needs to combine this operator with a :class:`FieldZeroPadder` or even
    two (one for the energy and one for the spatial subdomain)
    """
    
    amps = [amplitudes,] if isinstance(amplitudes,ift.Operator) else amplitudes
    
    tgt = ift.DomainTuple.make(target)
    if len(tgt) != len(amps):
        raise ValueError
    stdmean, stdstd = float(stdmean), float(stdstd)
    if stdstd <= 0:
        raise ValueError
    
    psp = [aa.target[0] for aa in amplitudes]
    hsp = ift.DomainTuple.make([tt.get_default_codomain() for tt in tgt])
    
    ht = ift.HarmonicTransformOperator(hsp, target=tgt[0], space=0)
    pd = ift.PowerDistributor(hsp, psp[0], 0)
    
    for i in range(1,len(amps)):
        ht = ift.HarmonicTransformOperator(ht.target,target=tgt[i], space=i) @ ht
        pd = pd @ ift.PowerDistributor(pd.domain, psp[i], space = i)
    
    spaces = tuple(range(len(amps)))

    a = ift.ContractionOperator(pd.domain, spaces[1:]).adjoint @ amps[0]
    for i in range(1,len(amps)):
        a = a * (ift.ContractionOperator(pd.domain,
                 spaces[:i] + spaces[(i+1):]).adjoint @ amps[i])
    
    expander = ift.VdotOperator(ift.full(a.target,1.)).adjoint

    Std = stdstd * ift.ducktape(expander.domain, None, names[1])
    Std = expander @ (ift.Adder(ift.full(expander.domain,stdmean))@Std).exp()
    
    A = pd @ (Std * a)
    # For `vol` see comment in `CorrelatedField`
    vol = reduce(mul, [sp.scalar_dvol**-0.5 for sp in hsp])
    return ht(vol*A*ift.ducktape(hsp, None, names[0]))

if __name__ == '__main__':
    import numpy as np
    from normalized_amplitude import NormalizedAmplitude
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
        a.append(NormalizedAmplitude(target, 16, 1, 1, -3, 1, 0, 1, 0, 1))
    tgt = ift.makeDomain(tuple(spaces))
    op = CorrelatedFieldNormAmplitude(tgt,a,0.,1.)
    fld = ift.from_random('normal', op.domain)
    ift.extra.check_jacobian_consistency(op, fld)
    