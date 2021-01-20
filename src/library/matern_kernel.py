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
# Authors: Matteo Guardiani, Jakob Roth
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from functools import reduce
from operator import mul


from ..field import Field
from ..multi_field import MultiField
from ..domains.power_space import PowerSpace
from ..operators.operator import Operator
from ..operators.harmonic_operators import HarmonicTransformOperator
from ..operators.simple_linear_operators import ducktape, VdotOperator
from ..operators.distributors import PowerDistributor
from ..operators.contraction_operator import ContractionOperator
from ..operators.adder import Adder
from ..operators.normal_operators import NormalTransform, LognormalTransform
from ..sugar import full, makeDomain, makeField


class MaternKernelMaker:
    """Construction helper for hierarchical matern kernel correlated field
    models.

    The matern kernel field models are parametrized by creating
    matern kernel power spectrum operators ("amplitudes") via calls to
    :func:`add_fluctuations` that act on the targeted field subdomains.
    During creation of the :class:`MaternKernelMaker` via
    :func:`make`, a global offset from zero of the field model
    can be defined and an operator applying fluctuations
    around this offset is parametrized.

    The resulting field model operator has a
    :class:`~nifty7.multi_domain.MultiDomain` as its domain and
    expects its input values to be univariately gaussian.

    The target of the constructed operator will be a
    :class:`~nifty7.domain_tuple.DomainTuple` containing the
    `target_subdomains` of the added fluctuations in the order of
    the `add_fluctuations` calls.

    Creation of the model operator is completed by calling the method
    :func:`finalize`, which returns the configured operator.

    See the methods :func:`make`, :func:`add_fluctuations`
    and :func:`finalize` for further usage information."""
    def __init__(self, offset_mean, offset_fluctuations_op, prefix):
        if not isinstance(offset_fluctuations_op, Operator):
            raise TypeError("offset_fluctuations_op needs to be an operator")
        self._a = []
        self._target_subdomains = []

        self._offset_mean = offset_mean
        self._azm = offset_fluctuations_op
        self._prefix = prefix

    @staticmethod
    def make(offset_mean, offset_std_mean, offset_std_std, prefix):
        """Returns a MaternKernelMaker object.

        Parameters
        ----------
        offset_mean : float
            Mean offset from zero of the matern kernel field to be made.
        offset_std : tuple of float
            Mean standard deviation and standard deviation of the standard
            deviation of the offset. No, this is not a word duplication.
        prefix : string
            Prefix to the names of the domains of the cf operator to be made.
            This determines the names of the operator domain.
        """
        zm = LognormalTransform(offset_std_mean, offset_std_std,
                                prefix + 'zeromode', 0)
        return MaternKernelMaker(offset_mean, zm, prefix)

    def add_fluctuations(self,
                         target_subdomain,
                         a,
                         b,
                         c,
                         prefix=''):
        """Function to add matern kernels to the field to be made.

        The matern kernel amplitude is parametrized in the following way:
        .. math ::
            E(f) = \\frac{a}{\\left(1 + \\left(\\frac{k}{b}\\right)^2\\right)^c}
            
        With a being the scale, b the cutoff and c half the slope of the
        power law
        
        Parameters
        ----------
        target_subdomain : :class:`~nifty7.domain.Domain`, \
                           :class:`~nifty7.domain_tuple.DomainTuple`
            Target subdomain on which the correlation structure defined
            in this call should hold.
        scale : tuple of float (mean, std)

        cutoff : tuple of float (mean, std)

        halfslope: tuple of float (mean, std)

        """
        harmonic_partner = target_subdomain.get_default_codomain()
        psp = PowerSpace(harmonic_partner)
        target_subdomain = makeDomain(target_subdomain)
        
        pref = LognormalTransform(*a,
                                   self._prefix + prefix + 'scale', 0)
        modpref = LognormalTransform(*b,
                                 self._prefix + prefix + 'cutoff', 0)
        loglogsqslope = NormalTransform(*c,
                                self._prefix + prefix + 'halfslope', 0)
        
        expander = VdotOperator(full(psp,1.)).adjoint
        k_squared = makeField(psp, psp.k_lengths**2)

        a = expander @ pref.log() # FIX ME: look for nicer implementation, if any
        b = VdotOperator(k_squared).adjoint @ modpref.power(-2.)
        c = expander @ loglogsqslope

        ker = Adder(full(psp, 1.)) @ b
        ker = c * ker.log() + a
        amp = ker.exp()

        self._a.append(amp)
        self._target_subdomains.append(target_subdomain)

    def finalize(self):
        """Finishes model construction process and returns the constructed
        operator.
        """
        n_amplitudes = len(self._a)
        
        hspace = makeDomain(
            [dd.target[0].harmonic_partner for dd in self._a])
        spaces = tuple(range(n_amplitudes))
        amp_space = 0

        expander = ContractionOperator(hspace, spaces=spaces).adjoint
        azm = expander @ self._azm

        ht = HarmonicTransformOperator(hspace,
                                       self._target_subdomains[0][amp_space],
                                       space=spaces[0])
        for i in range(1, n_amplitudes):
            ht = HarmonicTransformOperator(ht.target,
                                           self._target_subdomains[i][amp_space],
                                           space=spaces[i]) @ ht
        a = []
        for ii in range(n_amplitudes):
            co = ContractionOperator(hspace, spaces[:ii] + spaces[ii + 1:])
            pp = self._a[ii].target[amp_space]
            pd = PowerDistributor(co.target, pp, amp_space)
            a.append(co.adjoint @ pd @ self._a[ii])
        corr = reduce(mul, a)
        op = ht(azm*corr*ducktape(hspace, None, self._prefix + 'xi'))

        if self._offset_mean is not None:
            offset = self._offset_mean
            # Deviations from this offset must not be considered here as they
            # are learned by the zeromode
            if isinstance(offset, (Field, MultiField)):
                op = Adder(offset) @ op
            else:
                offset = float(offset)
                op = Adder(full(op.target, offset)) @ op
        return op
        
        
    @property
    def amplitude(self):
        """Analoguous to :func:`~nifty7.library.correlated_fields.CorrelatedFieldMaker.amplitude`."""
        return self._a

    @property
    def power_spectrum(self):
        """Analoguous to :func:`~nifty7.library.correlated_fields.CorrelatedFieldMaker.power_spectrum`."""
        return self.amplitude**2
