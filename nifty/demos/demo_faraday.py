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

    NIFTY demo applying transformations to the "Faraday Map" [#]_.

    References
    ----------
    .. [#] N. Opermann et. al., "An improved map of the Galactic Faraday sky",
        Astronomy & Astrophysics, Volume 542, id.A93, 06/2012;
        `arXiv:1111.6186 <http://www.arxiv.org/abs/1111.6186>`_

"""
from __future__ import division
from nifty import *



##-----------------------------------------------------------------------------

# (global) Faraday map
m = field(HPSpace(128), val=np.load(os.path.join(get_demo_dir(),
                                           "demo_faraday_map.npy")))

##-----------------------------------------------------------------------------

##=============================================================================

def run(projection=False, power=False):
    """
        Runs the demo.

        Parameters
        ----------
        projection : bool, *optional*
            Whether to additionaly show projections or not (default: False).
        power : bool, *optional*
            Whether to additionaly show power spectra or not (default: False).

    """
    nicely = {"vmin":-4, "vmax":4, "cmap":ncmap.fm()}

    # (a) representation on HEALPix grid
    m0 = m
    m0.plot(title=r"$m$ on a HEALPix grid", **nicely)
    nicely.update({"cmap":ncmap.fm()}) # healpy bug workaround

    # (b) representation in spherical harmonics
    k_space = m0.target # == lm_space(383, mmax=383)
    m1 = m0.transform(k_space) # == m.transform()
#    m1.plot(title=r"$m$ in spherical harmonics")

    if(power):
        m1.plot(title=r"angular power spectrum of $m$", vmin=1E-2, vmax=1E+1, mono=False)
    if(projection):
        # define projection operator
        Sk = projection_operator(m1.domain)
        # project quadrupole
        m2 = Sk(m0, band=2)
        m2.plot(title=r"angular quadrupole of $m$ on a HEALPix grid", **nicely)

    # (c) representation on Gauss-Legendre grid
    y_space = m.target.get_codomain(coname="gl") # == gl_space(384, nlon=767)
    m3 = m1.transform(y_space) # == m0.transform().transform(y_space)
    m3.plot(title=r"$m$ on a spherical Gauss-Legendre grid", **nicely)

    if(projection):
        m4 = Sk(m3, band=2)
        m4.plot(title=r"angular quadrupole of $m$ on a Gauss-Legendre grid", **nicely)

    # (d) representation on regular grid
    y_space = GLSpace(384, nlon=768) # auxiliary gl_space
    z_space = RGSpace([768, 384], dist=[1/360, 1/180])
    m5 = m1.transform(y_space)
    m5.cast_domain(z_space)
    m5.set_val(np.roll(m5.val[::-1, ::-1], y_space.paradict['nlon']//2, axis=1)) # rearrange value array
    m5.plot(title=r"$m$ on a regular 2D grid", **nicely)

    if(power):
        m5.target.set_power_indices(log=False)
        m5.plot(power=True, title=r"Fourier power spectrum of $m$", vmin=1E-3, vmax=1E+0, mono=False)
    if(projection):
        m5.target.set_power_indices(log=False)
        # define projection operator
        Sk = projection_operator(m5.target)
        # project quadrupole
        m6 = Sk(m5, band=2)
        m6.plot(title=r"Fourier quadrupole of $m$ on a regular 2D grid", **nicely)

##=============================================================================

##-----------------------------------------------------------------------------

if(__name__=="__main__"):
#    pl.close("all")

    # run demo
    run(projection=False, power=False)
    # define projection operator
    Sk = projection_operator(m.target)

##-----------------------------------------------------------------------------

