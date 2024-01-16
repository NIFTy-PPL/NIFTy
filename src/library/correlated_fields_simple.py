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
# Copyright(C) 2013-2020 Max-Planck-Society
# Author: Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from warnings import warn

import numpy as np

from ..domain_tuple import DomainTuple
from ..domains.power_space import PowerSpace
from ..operators.adder import Adder
from ..operators.contraction_operator import ContractionOperator
from ..operators.distributors import PowerDistributor
from ..operators.harmonic_operators import HarmonicTransformOperator
from ..operators.normal_operators import LognormalTransform, NormalTransform
from ..operators.simple_linear_operators import ducktape
from ..operators.value_inserter import ValueInserter
from ..sugar import full, makeField, makeOp
from .correlated_fields import (_log_vol, _Normalization,
                                _relative_log_k_lengths, _SlopeRemover,
                                _TwoLogIntegrations)


def SimpleCorrelatedField(
    target,
    offset_mean,
    offset_std,
    fluctuations,
    flexibility,
    asperity,
    loglogavgslope,
    prefix="",
    harmonic_partner=None,
):
    """Simplified version of :class:`~nifty8.library.correlated_fields.CorrelatedFieldMaker`.

    Assumes `total_N = 0`, `dofdex = None` and the presence of only one power
    spectrum, i.e. only one call of
    :func:`~nifty8.library.correlated_fields.CorrelatedFieldMaker.add_fluctuations`.

    See also
    --------
    * The simple correlated field model has first been described in "Comparison
      of classical and Bayesian imaging in radio interferometry", A&A 646, A84
      (2021) by P. Arras et al.
      `<https://doi.org/10.1051/0004-6361/202039258>`_

    Consider citing this paper, if you use the simple correlated field model.
    """
    target = DomainTuple.make(target)
    if len(target) != 1:
        raise ValueError
    target = target[0]
    if harmonic_partner is None:
        harmonic_partner = target.get_default_codomain()
    else:
        target.check_codomain(harmonic_partner)
        harmonic_partner.check_codomain(target)
    for kk in (fluctuations, loglogavgslope):
        if len(kk) != 2:
            raise TypeError
    for kk in (offset_std, flexibility, asperity):
        if not (kk is None or len(kk) == 2):
            raise TypeError
    if flexibility is None and asperity is not None:
        raise ValueError
    fluct = LognormalTransform(*fluctuations, prefix + 'fluctuations', 0)
    avgsl = NormalTransform(*loglogavgslope, prefix + 'loglogavgslope', 0)

    pspace = PowerSpace(harmonic_partner)
    twolog = _TwoLogIntegrations(pspace)
    expander = ContractionOperator(twolog.domain, 0).adjoint
    ps_expander = ContractionOperator(pspace, 0).adjoint
    vslope = makeOp(makeField(pspace, _relative_log_k_lengths(pspace)))
    slope = vslope @ ps_expander @ avgsl
    a = slope

    if flexibility is not None:
        flex = LognormalTransform(*flexibility, prefix + 'flexibility', 0)
        dom = twolog.domain[0]
        vflex = np.empty(dom.shape)
        vflex[0] = vflex[1] = np.sqrt(_log_vol(pspace))
        vflex = makeOp(makeField(dom, vflex))
        sig_flex = vflex @ expander @ flex
        xi = ducktape(dom, None, prefix + 'spectrum')

        shift = np.empty(dom.shape)
        shift[0] = _log_vol(pspace)**2 / 12.
        shift[1] = 1
        shift = makeField(dom, shift)
        if asperity is None:
            asp = makeOp(shift.ptw("sqrt")) @ (xi*sig_flex)
        else:
            asp = LognormalTransform(*asperity, prefix + 'asperity', 0)
            vasp = np.empty(dom.shape)
            vasp[0] = 1
            vasp[1] = 0
            vasp = makeOp(makeField(dom, vasp))
            sig_asp = vasp @ expander @ asp
            asp = xi*sig_flex*(Adder(shift) @ sig_asp).ptw("sqrt")
        a = a + _SlopeRemover(pspace, 0) @ twolog @ asp
    a = _Normalization(pspace, 0) @ a
    maskzm = np.ones(pspace.shape)
    maskzm[0] = 0
    maskzm = makeOp(makeField(pspace, maskzm))
    a = (maskzm @ ((ps_expander @ fluct)*a))
    if offset_std is not None:
        zm = LognormalTransform(*offset_std, prefix + 'zeromode', 0)
        insert = ValueInserter(pspace, (0,))
        a = a + insert(zm)
    a = a.scale(target.total_volume)

    ht = HarmonicTransformOperator(harmonic_partner, target)
    pd = PowerDistributor(harmonic_partner, pspace)
    xi = ducktape(harmonic_partner, None, prefix + 'xi')
    op = ht(pd(a)*xi)
    if offset_mean is not None:
        op = Adder(full(op.target, float(offset_mean))) @ op
    op.amplitude = a
    op.power_spectrum = a**2

    try:
        from .. import RGSpace
        from .. import re as jft

        if not all(isinstance(dom, RGSpace) for dom in op.target):
            # warn(f"unable to add JAX operator for {op.target!r}")
            raise ImportError("short-circuit JAX init")

        dists = tuple(e for di in op.target for e in di.distances)
        cfm = jft.CorrelatedFieldMaker(prefix=prefix)
        cfm.add_fluctuations(
            shape=op.target.shape,
            distances=dists,
            fluctuations=fluctuations,
            loglogavgslope=loglogavgslope,
            flexibility=flexibility,
            asperity=asperity,
            prefix="",
            harmonic_type="fourier",
            non_parametric_kind="power",
        )
        cfm.set_amplitude_total_offset(
            offset_mean=offset_mean, offset_std=offset_std
        )
        cf = cfm.finalize()

        op._jax_expr = cf
        op.amplitude._jax_expr = cfm.amplitude
        op.power_spectrum._jax_expr = cfm.power_spectrum
    except (ImportError, TypeError) as e:
        pass
        # if isinstance(e, TypeError):
        #     warn(f"no JAX operator for this configuration;\n{e}")

    return op
