## NIFTY (Numerical Information Field Theory) has been developed at the
## Max-Planck-Institute for Astrophysics.
##
## Copyright (C) 2013 Max-Planck-Society
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

"""
    ..                  __   ____   __
    ..                /__/ /   _/ /  /_
    ..      __ ___    __  /  /_  /   _/  __   __
    ..    /   _   | /  / /   _/ /  /   /  / /  /
    ..   /  / /  / /  / /  /   /  /_  /  /_/  /
    ..  /__/ /__/ /__/ /__/    \___/  \___   /  demo
    ..                               /______/

    NIFTY demo for (extended) critical Wiener filtering of Gaussian random signals.

"""
from __future__ import division
import matplotlib as mpl
mpl.use('Agg')

import imp
#nifty = imp.load_module('nifty', None,
#                        '/home/steininger/Downloads/nifty', ('','',5))

from nifty import *


##=============================================================================

class problem(object):

    def __init__(self, x_space, s2n=12, **kwargs):
        """
            Sets up a Wiener filter problem.

            Parameters
            ----------
            x_space : space
                Position space the signal lives in.
            s2n : float, *optional*
                Signal-to-noise ratio (default: 2).

        """
        self.store = []
        ## set signal space
        self.z = x_space
        ## set conjugate space
        self.k = self.z.get_codomain()
        #self.k.power_indices.set_default()
        #self.k.set_power_indices(**kwargs)

        ## set some power spectrum
        self.power = (lambda k: 42 / (k + 1) ** 3)

        ## define signal covariance
        self.S = power_operator(self.k, spec=self.power, bare=True)
        ## define projector to spectral bands
        self.Sk = self.S.get_projection_operator()
        ## generate signal
        self.s = self.S.get_random_field(domain=self.z)

        ## define response
        self.R = response_operator(self.z, sigma=0.0, mask=1.0)
        ## get data space
        d_space = self.R.target

        ## define noise covariance
        #self.N = diagonal_operator(d_space, diag=abs(s2n) * self.s.var(), bare=True)
        self.N = diagonal_operator(d_space, diag=abs(s2n), bare=True)
        ## define (plain) projector
        self.Nj = projection_operator(d_space)
        ## generate noise
        n = self.N.get_random_field(domain=d_space)

        ## compute data
        self.d = self.R(self.s) + n

        ## define information source
        #self.j = self.R.adjoint_times(self.N.inverse_times(self.d), target=self.k)
        self.j = self.R.adjoint_times(self.N.inverse_times(self.d))
        ## define information propagator
        self.D = propagator_operator(S=self.S,
                                     N=self.N,
                                     R=self.R)

        ## reserve map
        self.m = None

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def solve(self, newspec=None):
        """
            Solves the Wiener filter problem for a given power spectrum
            reconstructing a signal estimate.

            Parameters
            ----------
            newspace : {scalar, list, array, field, function}, *optional*
                Assumed power spectrum (default: k ** -2).

        """
        ## set (given) power spectrum
        if(newspec is None):
            newspec = np.r_[1, 1 / self.k.power_indices["kindex"][1:] ** 2] ## Laplacian
        elif(newspec is False):
            newspec = self.power ## assumed to be known
        self.S.set_power(newspec, bare=True)

        ## reconstruct map
        self.m = self.D(self.j, W=self.S, tol=1E-3, note=False)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def solve_critical(self, newspec=None, q=0, alpha=1, delta=1, epsilon=0):
        """
            Solves the (generalised) Wiener filter problem
            reconstructing a signal estimate and a power spectrum.

            Parameters
            ----------
            newspace : {scalar, list, array, field, function}, *optional*
                Initial power spectrum (default: k ** -2).
            q : {scalar, list, array}, *optional*
                Spectral scale parameter of the assumed inverse-Gamme prior
                (default: 0).
            alpha : {scalar, list, array}, *optional*
                Spectral shape parameter of the assumed inverse-Gamme prior
                (default: 1).
            delta : float, *optional*
                First filter perception parameter (default: 1).
            epsilon : float, *optional*
                Second filter perception parameter (default: 0).

            See Also
            --------
            infer_power

        """
        ## set (initial) power spectrum
        if(newspec is None):
            newspec = np.r_[1, 1 / self.k.power_indices["kindex"][1:] ** 2] ## Laplacian
        elif(newspec is False):
            newspec = self.power ## assumed to be known
        self.S.set_power(newspec, bare=True)

        ## pre-compute denominator
        denominator = self.k.power_indices["rho"] + 2 * (alpha - 1 + abs(epsilon))

        self.save_signal_and_data()

        ## iterate
        i = 0
        iterating = True
        while(iterating):

            ## reconstruct map
            self.m = self.D(self.j, W=self.S, tol=1E-3, note=True)
            if(self.m is None):
                break
            #print'Reconstructed m'
            ## reconstruct power spectrum
            tr_B1 = self.Sk.pseudo_tr(self.m) ## == Sk(m).pseudo_dot(m)
            print 'Calculated trace B1'
            print ('tr_b1', tr_B1)
            tr_B2 = self.Sk.pseudo_tr(self.D, loop=True)
            print 'Calculated trace B2'
            print ('tr_B2', tr_B2)
            numerator = 2 * q + tr_B1 +  tr_B2 * abs(delta)  ## non-bare(!)
            power = numerator / denominator
            print ('numerator', numerator)
            print ('denominator', denominator)
            print ('power', power)
            print 'Calculated power'

            #power = np.clip(power, 0.00000001, np.max(power))
            self.store += [{'tr_B1': tr_B1,
                            'tr_B2': tr_B2,
                            'num': numerator,
                            'denom': denominator}]
            ## check convergence
            dtau = log(power / self.S.get_power(), base=self.S.get_power())
            print ('dtau', np.max(np.abs(dtau)))
            iterating = (np.max(np.abs(dtau)) > 2E-2)
            #printmax(np.abs(dtau))
            self.save_map(i)
            i += 1
            ## update signal covariance
            self.S.set_power(power, bare=False) ## auto-updates D

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def save_signal_and_data(self):
        self.s.plot(title="signal", save="img/signal.png")

        try:
            d_ = field(self.z, val=self.d.val, target=self.k)
            d_.plot(title="data", vmin=self.s.min(), vmax=self.s.max(),
                    save="img/data.png")
        except:
            pass


    def save_map(self, index=0):

        # save map
        if(self.m is None):
            pass
        else:
            self.m.plot(title="reconstructed map",
                        vmin=self.s.min(), vmax=self.s.max(),
                        save="img/map_"+str(index)+".png")
            self.m.plot(power=True, mono=False,
                        other=(self.power, self.S.get_power()),
                        nbin=None, binbounds=None, log=False,
                        save='img/map_power_'+str(index)+".png")

    def plot(self):
        """
            Produces plots.

        """
        ## plot signal
        self.s.plot(title="signal")
        ## plot data
        try:
            d_ = field(self.z, val=self.d.val, target=self.k)
            d_.plot(title="data", vmin=self.s.min(), vmax=self.s.max())
        except:
            pass
        ## plot map
        if(self.m is None):
            self.s.plot(power=True, mono=False, other=self.power)
        else:
            self.m.plot(title="reconstructed map", vmin=self.s.min(), vmax=self.s.max())
            self.m.plot(power=True, mono=False, other=(self.power, self.S.get_power()))

##=============================================================================

##-----------------------------------------------------------------------------
#
if(__name__=="__main__"):
    x = rg_space((128), zerocenter=True)
    p = problem(x, log = False)
    about.warnings.off()
##    pl.close("all")
#
#    ## define signal space
#    x_space = rg_space(128)
#
#    ## setup problem
#    p = problem(x_space, log=True)
#    ## solve problem given some power spectrum
#    p.solve()
#    ## solve problem
#    p.solve_critical()
#
#    p.plot()
#
#    ## retrieve objects
#    k_space = p.k
#    power = p.power
#    S = p.S
#    Sk = p.Sk
#    s = p.s
#    R = p.R
#    d_space = p.R.target
#    N = p.N
#    Nj = p.Nj
#    d = p.d
#    j = p.j
#    D = p.D
#    m = p.m

##-----------------------------------------------------------------------------

