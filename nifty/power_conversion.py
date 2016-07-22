## NIFTY (Numerical Information Field Theory) has been developed at the
## Max-Planck-Institute for Astrophysics.
##
## Copyright (C) 2015 Max-Planck-Society
##
## Author: Marco Selig
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

import numpy as np
from numpy import pi
from nifty.config import about
from nifty.field import Field
from nifty.nifty_simple_math import sqrt, exp, log


def power_backward_conversion_lm(k_space, p, mean=None):
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
    C_0_Omega = Field(k_space,val=0)
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
    C_0_Omega = Field(k_space,val=0)
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


def power_backward_conversion_rg(k_space, p, mean=None, bare=True):
    """
        This function is designed to convert a theoretical/statistical power
        spectrum of a log-normal field to the theoretical power spectrum of
        the underlying Gaussian field.
        The function only works for power spectra defined for rg_spaces

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
            spectrum (default: None).
            WARNING: a mean that is too low can violate positive definiteness
            of the log-normal field. In this case the function produces an
            error.
        bare : bool, *optional*
            whether `p` is the bare power spectrum or not (default: True).

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
        .. [#] M. Greiner and T.A. Ensslin, "Log-transforming the matter
               power spectrum";
            `arXiv:1312.1354 <http://arxiv.org/abs/1312.1354>`_
    """

    pindex = k_space.power_indices['pindex']
    weight = k_space.get_weight()

    monopole_index = pindex.argmin()

    # Cast the supplied spectrum
    spec = k_space.enforce_power(p)
    # Now we mimick the weightning behaviour of
    # spec = power_operator(k_space,spec=p,bare=bare).get_power(bare=False)
    # by appliying the weight from the k_space
    if bare:
        spec *= weight

    #TODO: Does this realy set the mean to the monopole as promised in the docs? -> Check!
    if mean is None:
        mean = 0.
    else:
        spec[0] = 0.

    p_val = pindex.apply_scalar_function(lambda x: spec[x],
                                         dtype=spec.dtype.type)
    power_field = Field(k_space, val=p_val, zerocenter=True).transform()
    power_field += (mean**2)

    if power_field.min() < 0:
        raise ValueError(about._errors.cstring(
            "ERROR: spectrum or mean incompatible with positive " +
            "definiteness. \n Try increasing the mean."))

    log_of_power_field = power_field.apply_scalar_function(np.log,
                                                           inplace=True)
    power_spectrum_1 = log_of_power_field.power()**(0.5)
    power_spectrum_1[0] = log_of_power_field.transform()[monopole_index]

    power_spectrum_0 = k_space.calc_weight(p_val).sum() + (mean**2)
    power_spectrum_0 = np.log(power_spectrum_0)
    power_spectrum_0 *= (0.5 / weight)

    log_mean = weight * (power_spectrum_1[0] - power_spectrum_0)

    power_spectrum_1[0] = 0.

    # Mimik
    # power_operator(k_space,spec=power_spectrum_1,bare=False).\
    #  get_power(bare=True).real
    if bare:
        power_spectrum_1 /= weight

    return log_mean.real, power_spectrum_1.real


def power_forward_conversion_rg(k_space, p, mean=0, bare=True):
    """
        This function is designed to convert a theoretical/statistical power
        spectrum of a Gaussian field to the theoretical power spectrum of
        the exponentiated field.
        The function only works for power spectra defined for rg_spaces

        Parameters
        ----------
        k_space : nifty.rg_space,
            a regular grid space with the attribute `Fourier = True`
        p : np.array,
            the power spectrum of the Gaussian field.
            Needs to have the same number of entries as
            `k_space.get_power_indices()[0]`
        mean : float, *optional*
            specifies the mean of the Gaussian field (default: 0).
        bare : bool, *optional*
            whether `p` is the bare power spectrum or not (default: True).

        Returns
        -------
        p1 : np.array,
            the power spectrum of the exponentiated Gaussian field.

        References
        ----------
        .. [#] M. Greiner and T.A. Ensslin,
            "Log-transforming the matter power spectrum";
            `arXiv:1312.1354 <http://arxiv.org/abs/1312.1354>`_
    """

    pindex = k_space.power_indices['pindex']
    weight = k_space.get_weight()
    # Cast the supplied spectrum
    spec = k_space.enforce_power(p)
    # Now we mimick the weightning behaviour of
    # spec = power_operator(k_space,spec=p,bare=bare).get_power(bare=False)
    # by appliying the weight from the k_space
    if bare:
        spec *= weight

    S_val = pindex.apply_scalar_function(lambda x: spec[x],
                                         dtype=spec.dtype.type)

    # S_x is a field
    S_x = Field(k_space, val=S_val, zerocenter=True).transform()
    # s_0 is a scalar
    s_0 = k_space.calc_weight(S_val, power=1).sum()

    # Calculate the new power_field
    S_x += s_0
    S_x += 2*mean

    power_field = S_x.apply_scalar_function(np.exp, inplace=True)

    new_spec = power_field.power()**(0.5)

    # Mimik
    # power_operator(k_space,spec=p1,bare=False).get_power(bare=True).real
    if bare:
        new_spec /= weight

    return new_spec.real