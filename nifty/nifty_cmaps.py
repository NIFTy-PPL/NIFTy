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
    ..  /__/ /__/ /__/ /__/    \___/  \___   /  cmaps
    ..                               /______/

    This module provides the `ncmap` class whose static methods return color
    maps.

    The visualization of fields is useful for obvious reasons, and therefore
    some nice color maps are here to be found. Those are segmented color maps
    that can be used in many settings, including the native plotting method for
    fields. (Some of the color maps offered here are results from IFT
    publications, cf. references below.)

    Examples
    --------
    >>> from nifty.nifty_cmaps import *
    >>> f = field(rg_space([42, 42]), random="uni", vmin=-1)
    >>> f[21:] = f.smooth(sigma=1/42)[21:]
    >>> [f.plot(cmap=cc, vmin=-0.8, vmax=0.8) for cc in [None, ncmap.pm()]]
    ## two 2D plots open

"""
from __future__ import division
from matplotlib.colors import LinearSegmentedColormap as cm


##-----------------------------------------------------------------------------

class ncmap(object):
    """
        ..     __ ___    _______   __ ___ ____    ____ __   ______
        ..   /   _   | /   ____/ /   _    _   | /   _   / /   _   |
        ..  /  / /  / /  /____  /  / /  / /  / /  /_/  / /  /_/  /
        .. /__/ /__/  \______/ /__/ /__/ /__/  \______| /   ____/  class
        ..                                             /__/

        NIFTY support class for color maps.

        This class provides several *nifty* color maps that are returned by
        its static methods. The `ncmap` class is not meant to be initialised.

        See Also
        --------
        matplotlib.colors.LinearSegmentedColormap

        Examples
        --------
        >>> f = field(rg_space([42, 42]), random="uni", vmin=-1)
        >>> f.plot(cmap=ncmap.pm(), vmin=-1, vmax=1)
        ## 2D plot opens

    """
    __init__ = None

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @staticmethod
    def he(ncolors=256):
        """
            Returns a color map often used in High Energy Astronomy.

            Parameters
            ----------
            ncolors : int, *optional*
                Number of color segments (default: 256).

            Returns
            -------
            cmap : matplotlib.colors.LinearSegmentedColormap instance
                Linear segmented color map.

        """
        segmentdata = {"red":   [(0.000, 0.0, 0.0), (0.167, 0.0, 0.0),
                                 (0.333, 0.5, 0.5), (0.500, 1.0, 1.0),
                                 (0.667, 1.0, 1.0), (0.833, 1.0, 1.0),
                                 (1.000, 1.0, 1.0)],
                       "green": [(0.000, 0.0, 0.0), (0.167, 0.0, 0.0),
                                 (0.333, 0.0, 0.0), (0.500, 0.0, 0.0),
                                 (0.667, 0.5, 0.5), (0.833, 1.0, 1.0),
                                 (1.000, 1.0, 1.0)],
                       "blue":  [(0.000, 0.0, 0.0), (0.167, 1.0, 1.0),
                                 (0.333, 0.5, 0.5), (0.500, 0.0, 0.0),
                                 (0.667, 0.0, 0.0), (0.833, 0.0, 0.0),
                                 (1.000, 1.0, 1.0)]}

        return cm("High Energy", segmentdata, N=int(ncolors), gamma=1.0)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @staticmethod
    def fm(ncolors=256):
        """
            Returns a color map used in reconstruction of the "Faraday Map".

            Parameters
            ----------
            ncolors : int, *optional*
                Number of color segments (default: 256).

            Returns
            -------
            cmap : matplotlib.colors.LinearSegmentedColormap instance
                Linear segmented color map.

            References
            ----------
            .. [#] N. Opermann et. al.,
                "An improved map of the Galactic Faraday sky",
                Astronomy & Astrophysics, Volume 542, id.A93, 06/2012;
                `arXiv:1111.6186 <http://www.arxiv.org/abs/1111.6186>`_

        """
        segmentdata = {"red":   [(0.000, 0.35, 0.35), (0.100, 0.40, 0.40),
                                 (0.200, 0.25, 0.25), (0.410, 0.47, 0.47),
                                 (0.500, 0.80, 0.80), (0.560, 0.96, 0.96),
                                 (0.590, 1.00, 1.00), (0.740, 0.80, 0.80),
                                 (0.800, 0.80, 0.80), (0.900, 0.50, 0.50),
                                 (1.000, 0.40, 0.40)],
                       "green": [(0.000, 0.00, 0.00), (0.200, 0.00, 0.00),
                                 (0.362, 0.88, 0.88), (0.500, 1.00, 1.00),
                                 (0.638, 0.88, 0.88), (0.800, 0.25, 0.25),
                                 (0.900, 0.30, 0.30), (1.000, 0.20, 0.20)],
                       "blue":  [(0.000, 0.35, 0.35), (0.100, 0.40, 0.40),
                                 (0.200, 0.80, 0.80), (0.260, 0.80, 0.80),
                                 (0.410, 1.00, 1.00), (0.440, 0.96, 0.96),
                                 (0.500, 0.80, 0.80), (0.590, 0.47, 0.47),
                                 (0.800, 0.00, 0.00), (1.000, 0.00, 0.00)]}

        return cm("Faraday Map", segmentdata, N=int(ncolors), gamma=1.0)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @staticmethod
    def fu(ncolors=256):
        """
            Returns a color map used for the "Faraday Map Uncertainty".

            Parameters
            ----------
            ncolors : int, *optional*
                Number of color segments (default: 256).

            Returns
            -------
            cmap : matplotlib.colors.LinearSegmentedColormap instance
                Linear segmented color map.

            References
            ----------
            .. [#] N. Opermann et. al.,
                "An improved map of the Galactic Faraday sky",
                Astronomy & Astrophysics, Volume 542, id.A93, 06/2012;
                `arXiv:1111.6186 <http://www.arxiv.org/abs/1111.6186>`_

        """
        segmentdata = {"red":   [(0.000, 1.00, 1.00), (0.100, 0.80, 0.80),
                                 (0.200, 0.65, 0.65), (0.410, 0.60, 0.60),
                                 (0.500, 0.70, 0.70), (0.560, 0.96, 0.96),
                                 (0.590, 1.00, 1.00), (0.740, 0.80, 0.80),
                                 (0.800, 0.80, 0.80), (0.900, 0.50, 0.50),
                                 (1.000, 0.40, 0.40)],
                       "green": [(0.000, 0.90, 0.90), (0.200, 0.65, 0.65),
                                 (0.362, 0.95, 0.95), (0.500, 1.00, 1.00),
                                 (0.638, 0.88, 0.88), (0.800, 0.25, 0.25),
                                 (0.900, 0.30, 0.30), (1.000, 0.20, 0.20)],
                       "blue":  [(0.000, 1.00, 1.00), (0.100, 0.80, 0.80),
                                 (0.200, 1.00, 1.00), (0.410, 1.00, 1.00),
                                 (0.440, 0.96, 0.96), (0.500, 0.70, 0.70),
                                 (0.590, 0.42, 0.42), (0.800, 0.00, 0.00),
                                 (1.000, 0.00, 0.00)]}

        return cm("Faraday Uncertainty", segmentdata, N=int(ncolors),
                  gamma=1.0)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @staticmethod
    def pm(ncolors=256):
        """
            Returns a color map useful for a zero-centerd range of values.

            Parameters
            ----------
            ncolors : int, *optional*
                Number of color segments (default: 256).

            Returns
            -------
            cmap : matplotlib.colors.LinearSegmentedColormap instance
                Linear segmented color map.

        """
        segmentdata = {"red":   [(0.0, 1.00, 1.00), (0.1, 0.96, 0.96),
                                 (0.2, 0.84, 0.84), (0.3, 0.64, 0.64),
                                 (0.4, 0.36, 0.36), (0.5, 0.00, 0.00),
                                 (0.6, 0.00, 0.00), (0.7, 0.00, 0.00),
                                 (0.8, 0.00, 0.00), (0.9, 0.00, 0.00),
                                 (1.0, 0.00, 0.00)],
                       "green": [(0.0, 0.50, 0.50), (0.1, 0.32, 0.32),
                                 (0.2, 0.18, 0.18), (0.3, 0.08, 0.08),
                                 (0.4, 0.02, 0.02), (0.5, 0.00, 0.00),
                                 (0.6, 0.02, 0.02), (0.7, 0.08, 0.08),
                                 (0.8, 0.18, 0.18), (0.9, 0.32, 0.32),
                                 (1.0, 0.50, 0.50)],
                       "blue":  [(0.0, 0.00, 0.00), (0.1, 0.00, 0.00),
                                 (0.2, 0.00, 0.00), (0.3, 0.00, 0.00),
                                 (0.4, 0.00, 0.00), (0.5, 0.00, 0.00),
                                 (0.6, 0.36, 0.36), (0.7, 0.64, 0.64),
                                 (0.8, 0.84, 0.84), (0.9, 0.96, 0.96),
                                 (1.0, 1.00, 1.00)]}

        return cm("Plus Minus", segmentdata, N=int(ncolors), gamma=1.0)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @staticmethod
    def planck(ncolors=256):
        """
            Returns a color map similar to the one used for the "Planck CMB Map".

            Parameters
            ----------
            ncolors : int, *optional*
                Number of color segments (default: 256).

            Returns
            -------
            cmap : matplotlib.colors.LinearSegmentedColormap instance
                Linear segmented color map.

        """
        segmentdata = {"red":   [(0.0, 0.00, 0.00), (0.1, 0.00, 0.00),
                                 (0.2, 0.00, 0.00), (0.3, 0.00, 0.00),
                                 (0.4, 0.00, 0.00), (0.5, 1.00, 1.00),
                                 (0.6, 1.00, 1.00), (0.7, 1.00, 1.00),
                                 (0.8, 0.83, 0.83), (0.9, 0.67, 0.67),
                                 (1.0, 0.50, 0.50)],
                       "green": [(0.0, 0.00, 0.00), (0.1, 0.00, 0.00),
                                 (0.2, 0.00, 0.00), (0.3, 0.30, 0.30),
                                 (0.4, 0.70, 0.70), (0.5, 1.00, 1.00),
                                 (0.6, 0.70, 0.70), (0.7, 0.30, 0.30),
                                 (0.8, 0.00, 0.00), (0.9, 0.00, 0.00),
                                 (1.0, 0.00, 0.00)],
                       "blue":  [(0.0, 0.50, 0.50), (0.1, 0.67, 0.67),
                                 (0.2, 0.83, 0.83), (0.3, 1.00, 1.00),
                                 (0.4, 1.00, 1.00), (0.5, 1.00, 1.00),
                                 (0.6, 0.00, 0.00), (0.7, 0.00, 0.00),
                                 (0.8, 0.00, 0.00), (0.9, 0.00, 0.00),
                                 (1.0, 0.00, 0.00)]}

        return cm("Planck-like", segmentdata, N=int(ncolors), gamma=1.0)

##-----------------------------------------------------------------------------
