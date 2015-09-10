## NIFTY (Numerical Information Field Theory) has been developed at the
## Max-Planck-Institute for Astrophysics.
##
## Copyright (C) 2014 Max-Planck-Society
##
## Author: Maksim Greiner, Marco Selig
## Project homepage: <http://www.mpa-garching.mpg.de/ift/nifty/>
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
## See the GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program. If not, see <http://www.gnu.org/licenses/>.

#from nifty import *
import numpy as np
from nifty.keepers import about
from nifty.nifty_core import pi, field
from nifty.nifty_simple_math import sqrt, exp, log


def power_backward_conversion_lm(k_space,p,mean=None):
    """
        This function is designed to convert a theoretical/statistical power
        spectrum of a log-normal field to the theoretical power spectrum of
        the underlying Gaussian field.
        The function only works for power spectra defined for lm_spaces

        Parameters
        ----------
        k_space : nifty.rg_space,
            a regular grid space with the attribute `Fourier = True`
        p : np.array,
            the power spectrum of the log-normal field.
            Needs to have the same number of entries as
            `k_space.get_power_indices()[0]`
        mean : float, *optional*
            specifies the mean of the log-normal field. If `mean` is not
            specified the function will use the monopole of the power spectrum.
            If it is specified the function will NOT use the monopole of the
            spectrum. (default: None)
            WARNING: a mean that is too low can violate positive definiteness
            of the log-normal field. In this case the function produces an
            error.

        Returns
        -------
        mean : float,
            the recovered mean of the underlying Gaussian distribution.
        p1 : np.array,
            the power spectrum of the underlying Gaussian field, where the
            monopole has been set to zero. Eventual monopole power has been
            shifted to the mean.

        References
        ----------
        .. [#] M. Greiner and T.A. Ensslin, "Log-transforming the matter power spectrum";
            `arXiv:1312.1354 <http://arxiv.org/abs/1312.1354>`_
    """

    p = np.copy(p)
    if(mean is not None):
        p[0] = 4*pi*mean**2

    klen = k_space.get_power_indices()[0]
    C_0_Omega = field(k_space,val=0)
    C_0_Omega.val[:len(klen)] = p*sqrt(2*klen+1)/sqrt(4*pi)
    C_0_Omega = C_0_Omega.transform()

    if(np.any(C_0_Omega.val<0.)):
        raise ValueError(about._errors.cstring("ERROR: spectrum or mean incompatible with positive definiteness.\n Try increasing the mean."))
        return None

    lC = log(C_0_Omega)

    Z = lC.transform()

    spec = Z.val[:len(klen)]

    mean = (spec[0]-0.5*sqrt(4*pi)*log((p*(2*klen+1)/(4*pi)).sum()))/sqrt(4*pi)

    spec[0] = 0.

    spec = spec*sqrt(4*pi)/sqrt(2*klen+1)

    spec = np.real(spec)

    if(np.any(spec<0.)):
        spec = spec*(spec>0.)
        about.warnings.cprint("WARNING: negative modes set to zero.")

    return mean.real,spec


def power_forward_conversion_lm(k_space,p,mean=0):
    """
        This function is designed to convert a theoretical/statistical power
        spectrum of a Gaussian field to the theoretical power spectrum of
        the exponentiated field.
        The function only works for power spectra defined for lm_spaces

        Parameters
        ----------
        k_space : nifty.rg_space,
            a regular grid space with the attribute `Fourier = True`
        p : np.array,
            the power spectrum of the Gaussian field.
            Needs to have the same number of entries as
            `k_space.get_power_indices()[0]`
        m : float, *optional*
            specifies the mean of the Gaussian field (default: 0).

        Returns
        -------
        p1 : np.array,
            the power spectrum of the exponentiated Gaussian field.

        References
        ----------
        .. [#] M. Greiner and T.A. Ensslin, "Log-transforming the matter power spectrum";
            `arXiv:1312.1354 <http://arxiv.org/abs/1312.1354>`_
    """
    m = mean
    klen = k_space.get_power_indices()[0]
    C_0_Omega = field(k_space,val=0)
    C_0_Omega.val[:len(klen)] = p*sqrt(2*klen+1)/sqrt(4*pi)
    C_0_Omega = C_0_Omega.transform()

    C_0_0 = (p*(2*klen+1)/(4*pi)).sum()

    exC = exp(C_0_Omega+C_0_0+2*m)

    Z = exC.transform()

    spec = Z.val[:len(klen)]

    spec = spec*sqrt(4*pi)/sqrt(2*klen+1)

    spec = np.real(spec)

    if(np.any(spec<0.)):
        spec = spec*(spec>0.)
        about.warnings.cprint("WARNING: negative modes set to zero.")

    return spec