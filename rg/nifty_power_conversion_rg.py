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
from nifty.nifty_about import about
from nifty.nifty_core import field                                          
from nifty.nifty_simple_math import sqrt,exp,log                                          
from nifty.operators import power_operator            
                  


def power_backward_conversion_rg(k_space,p,mean=None,bare=True):
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
        .. [#] M. Greiner and T.A. Ensslin, "Log-transforming the matter power spectrum";
            `arXiv:1312.1354 <http://arxiv.org/abs/1312.1354>`_
    """
    pindex = k_space.get_power_indices()[2]
    V = k_space.vol.prod()**(-1)

    mono_ind = np.where(pindex==0)

    spec = power_operator(k_space,spec=p,bare=bare).get_power(bare=False)

    if(mean is None):
        mean = 0.
    else:
        spec[0] = 0.

    pf = field(k_space,val=spec[pindex]).transform()+mean**2

    if(np.any(pf.val<0.)):
        raise ValueError(about._errors.cstring("ERROR: spectrum or mean incompatible with positive definiteness.\n Try increasing the mean."))
        return None

    p1 = sqrt(log(pf).power())

    p1[0] = (log(pf)).transform()[mono_ind][0]

    p2 = 0.5*V*log(k_space.calc_weight(spec[pindex],1).sum()+mean**2)

    logmean = 1/V * (p1[0]-p2)

    p1[0] = 0.

    if(np.any(p1<0.)):
        raise ValueError(about._errors.cstring("ERROR: spectrum or mean incompatible with positive definiteness.\n Try increasing the mean."))
        return None

    if(bare==True):
        return logmean.real,power_operator(k_space,spec=p1,bare=False).get_power(bare=True).real
    else:
        return logmean.real,p1.real


def power_forward_conversion_rg(k_space,p,mean=0,bare=True):
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
        .. [#] M. Greiner and T.A. Ensslin, "Log-transforming the matter power spectrum";
            `arXiv:1312.1354 <http://arxiv.org/abs/1312.1354>`_
    """

    pindex = k_space.get_power_indices()[2]

    spec = power_operator(k_space,spec=p,bare=bare).get_power(bare=False)

    S_x = field(k_space,val=spec[pindex]).transform()

    S_0 = k_space.calc_weight(spec[pindex],1).sum()

    pf = exp(S_x+S_0+2*mean)

    p1 = sqrt(pf.power())

    if(bare==True):
        return power_operator(k_space,spec=p1,bare=False).get_power(bare=True).real
    else:
        return p1.real

