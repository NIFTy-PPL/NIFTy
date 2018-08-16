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
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from __future__ import absolute_import, division, print_function

import os

import numpy as np

from .. import dobj
from ..compat import *
from ..domains.gl_space import GLSpace
from ..domains.hp_space import HPSpace
from ..domains.power_space import PowerSpace
from ..domains.rg_space import RGSpace
from ..field import Field

# relevant properties:
# - x/y size
# - x/y/z log
# - x/y/z min/max
# - colorbar/colormap
# - axis on/off
# - title
# - axis labels
# - labels


def _mollweide_helper(xsize):
    xsize = int(xsize)
    ysize = xsize//2
    res = np.full(shape=(ysize, xsize), fill_value=np.nan, dtype=np.float64)
    xc, yc = (xsize-1)*0.5, (ysize-1)*0.5
    u, v = np.meshgrid(np.arange(xsize), np.arange(ysize))
    u, v = 2*(u-xc)/(xc/1.02), (v-yc)/(yc/1.02)

    mask = np.where((u*u*0.25 + v*v) <= 1.)
    t1 = v[mask]
    theta = 0.5*np.pi-(
        np.arcsin(2/np.pi*(np.arcsin(t1) + t1*np.sqrt((1.-t1)*(1+t1)))))
    phi = -0.5*np.pi*u[mask]/np.maximum(np.sqrt((1-t1)*(1+t1)), 1e-6)
    phi = np.where(phi < 0, phi+2*np.pi, phi)
    return res, mask, theta, phi


def _find_closest(A, target):
    # A must be sorted
    idx = np.clip(A.searchsorted(target), 1, len(A)-1)
    idx -= target - A[idx-1] < A[idx] - target
    return idx


def _makeplot(name):
    import matplotlib.pyplot as plt
    if dobj.rank != 0:
        plt.close()
        return
    if name is None:
        plt.show()
        plt.close()
        return
    extension = os.path.splitext(name)[1]
    if extension in (".pdf", ".png"):
        plt.savefig(name)
        plt.close()
    else:
        raise ValueError("file format not understood")


def _limit_xy(**kwargs):
    import matplotlib.pyplot as plt
    x1, x2, y1, y2 = plt.axis()
    x1 = kwargs.pop("xmin", x1)
    x2 = kwargs.pop("xmax", x2)
    y1 = kwargs.pop("ymin", y1)
    y2 = kwargs.pop("ymax", y2)
    plt.axis((x1, x2, y1, y2))


def _register_cmaps():
    try:
        if _register_cmaps._cmaps_registered:
            return
    except AttributeError:
        _register_cmaps._cmaps_registered = True

    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.pyplot as plt
    planckcmap = {'red':   ((0., 0., 0.), (.4, 0., 0.), (.5, 1., 1.),
                            (.7, 1., 1.), (.8, .83, .83), (.9, .67, .67),
                            (1., .5, .5)),
                  'green': ((0., 0., 0.), (.2, 0., 0.), (.3, .3, .3),
                            (.4, .7, .7), (.5, 1., 1.), (.6, .7, .7),
                            (.7, .3, .3), (.8, 0., 0.), (1., 0., 0.)),
                  'blue':  ((0., .5, .5), (.1, .67, .67), (.2, .83, .83),
                            (.3, 1., 1.), (.5, 1., 1.), (.6, 0., 0.),
                            (1., 0., 0.))}
    he_cmap = {'red':   ((0., 0., 0.), (.167, 0., 0.), (.333, .5, .5),
                         (.5, 1., 1.), (1., 1., 1.)),
               'green': ((0., 0., 0.), (.5, 0., 0.), (.667, .5, .5),
                         (.833, 1., 1.), (1., 1., 1.)),
               'blue':  ((0., 0., 0.), (.167, 1., 1.), (.333, .5, .5),
                         (.5, 0., 0.), (1., 1., 1.))}
    fd_cmap = {'red':   ((0., .35, .35), (.1, .4, .4), (.2, .25, .25),
                         (.41, .47, .47), (.5, .8, .8), (.56, .96, .96),
                         (.59, 1., 1.), (.74, .8, .8), (.8, .8, .8),
                         (.9, .5, .5), (1., .4, .4)),
               'green': ((0., 0., 0.), (.2, 0., 0.), (.362, .88, .88),
                         (.5, 1., 1.), (.638, .88, .88), (.8, .25, .25),
                         (.9, .3, .3), (1., .2, .2)),
               'blue':  ((0., .35, .35), (.1, .4, .4), (.2, .8, .8),
                         (.26, .8, .8), (.41, 1., 1.), (.44, .96, .96),
                         (.5, .8, .8), (.59, .47, .47), (.8, 0., 0.),
                         (1., 0., 0.))}
    fdu_cmap = {'red':   ((0., 1., 1.), (0.1, .8, .8), (.2, .65, .65),
                          (.41, .6, .6), (.5, .7, .7), (.56, .96, .96),
                          (.59, 1., 1.), (.74, .8, .8), (.8, .8, .8),
                          (.9, .5, .5), (1., .4, .4)),
                'green': ((0., .9, .9), (.362, .95, .95), (.5, 1., 1.),
                          (.638, .88, .88), (.8, .25, .25), (.9, .3, .3),
                          (1., .2, .2)),
                'blue':  ((0., 1., 1.), (.1, .8, .8), (.2, 1., 1.),
                          (.41, 1., 1.), (.44, .96, .96), (.5, .7, .7),
                          (.59, .42, .42), (.8, 0., 0.), (1., 0., 0.))}
    pm_cmap = {'red':   ((0., 1., 1.), (.1, .96, .96), (.2, .84, .84),
                         (.3, .64, .64), (.4, .36, .36), (.5, 0., 0.),
                         (1., 0., 0.)),
               'green': ((0., .5, .5), (.1, .32, .32), (.2, .18, .18),
                         (.3, .8, .8),  (.4, .2, .2), (.5, 0., 0.),
                         (.6, .2, .2), (.7, .8, .8), (.8, .18, .18),
                         (.9, .32, .32), (1., .5, .5)),
               'blue':  ((0., 0., 0.), (.5, 0., 0.), (.6, .36, .36),
                         (.7, .64, .64), (.8, .84, .84), (.9, .96, .96),
                         (1., 1., 1.))}

    plt.register_cmap(cmap=LinearSegmentedColormap("Planck-like", planckcmap))
    plt.register_cmap(cmap=LinearSegmentedColormap("High Energy", he_cmap))
    plt.register_cmap(cmap=LinearSegmentedColormap("Faraday Map", fd_cmap))
    plt.register_cmap(cmap=LinearSegmentedColormap("Faraday Uncertainty",
                                                   fdu_cmap))
    plt.register_cmap(cmap=LinearSegmentedColormap("Plus Minus", pm_cmap))


def _plot(f, ax, **kwargs):
    import matplotlib.pyplot as plt
    _register_cmaps()
    if isinstance(f, Field):
        f = [f]
    if not isinstance(f, list):
        raise TypeError("incorrect data type")
    for i, fld in enumerate(f):
        if not isinstance(fld, Field):
            raise TypeError("incorrect data type")
        if i == 0:
            dom = fld.domain
            if len(dom) != 1:
                raise ValueError("input field must have exactly one domain")
        else:
            if fld.domain != dom:
                raise ValueError("domain mismatch")
            if not (isinstance(dom[0], PowerSpace) or
                    (isinstance(dom[0], RGSpace) and len(dom[0].shape) == 1)):
                raise ValueError("PowerSpace or 1D RGSpace required")

    label = kwargs.pop("label", None)
    if not isinstance(label, list):
        label = [label] * len(f)

    linewidth = kwargs.pop("linewidth", 1.)
    if not isinstance(linewidth, list):
        linewidth = [linewidth] * len(f)

    alpha = kwargs.pop("alpha", None)
    if not isinstance(alpha, list):
        alpha = [alpha] * len(f)

    foo = kwargs.pop("norm", None)
    norm = {} if foo is None else {'norm': foo}

    dom = dom[0]
    ax.set_title(kwargs.pop("title", ""))
    ax.set_xlabel(kwargs.pop("xlabel", ""))
    ax.set_ylabel(kwargs.pop("ylabel", ""))
    cmap = kwargs.pop("colormap", plt.rcParams['image.cmap'])
    if isinstance(dom, RGSpace):
        if len(dom.shape) == 1:
            npoints = dom.shape[0]
            dist = dom.distances[0]
            xcoord = np.arange(npoints, dtype=np.float64)*dist
            for i, fld in enumerate(f):
                ycoord = fld.to_global_data()
                plt.plot(xcoord, ycoord, label=label[i],
                         linewidth=linewidth[i], alpha=alpha[i])
            _limit_xy(**kwargs)
            if label != ([None]*len(f)):
                plt.legend()
            return
        elif len(dom.shape) == 2:
            nx, ny = dom.shape
            dx, dy = dom.distances
            im = ax.imshow(
                f[0].to_global_data().T, extent=[0, nx*dx, 0, ny*dy],
                vmin=kwargs.get("zmin"), vmax=kwargs.get("zmax"),
                cmap=cmap, origin="lower", **norm)
            # from mpl_toolkits.axes_grid1 import make_axes_locatable
            # divider = make_axes_locatable(ax)
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            # plt.colorbar(im,cax=cax)
            plt.colorbar(im)
            _limit_xy(**kwargs)
            return
    elif isinstance(dom, PowerSpace):
        plt.xscale('log')
        plt.yscale('log')
        xcoord = dom.k_lengths
        for i, fld in enumerate(f):
            ycoord = fld.to_global_data()
            plt.plot(xcoord, ycoord, label=label[i],
                     linewidth=linewidth[i], alpha=alpha[i])
        _limit_xy(**kwargs)
        if label != ([None]*len(f)):
            plt.legend()
        return
    elif isinstance(dom, (HPSpace, GLSpace)):
        import pyHealpix
        xsize = 800
        res, mask, theta, phi = _mollweide_helper(xsize)
        if isinstance(dom, HPSpace):
            ptg = np.empty((phi.size, 2), dtype=np.float64)
            ptg[:, 0] = theta
            ptg[:, 1] = phi
            base = pyHealpix.Healpix_Base(int(np.sqrt(f[0].size//12)), "RING")
            res[mask] = f[0].to_global_data()[base.ang2pix(ptg)]
        else:
            ra = np.linspace(0, 2*np.pi, dom.nlon+1)
            dec = pyHealpix.GL_thetas(dom.nlat)
            ilat = _find_closest(dec, theta)
            ilon = _find_closest(ra, phi)
            ilon = np.where(ilon == dom.nlon, 0, ilon)
            res[mask] = f[0].to_global_data()[ilat*dom.nlon + ilon]
        plt.axis('off')
        plt.imshow(res, vmin=kwargs.get("zmin"), vmax=kwargs.get("zmax"),
                   cmap=cmap, origin="lower")
        plt.colorbar(orientation="horizontal")
        return

    raise ValueError("Field type not(yet) supported")


_plots = []
_kwargs = []


def plot(f, **kwargs):
    """Add a figure to the current list of plots.

    Notes
    -----
    After doing one or more calls `plot()`, one also needs to call
    `plot_finish()` to output the result.

    Parameters
    ----------
    f: Field, or list of Field objects
        If `f` is a single Field, it must live over a single `RGSpace`,
        `PowerSpace`, `HPSpace`, `GLSPace`.
        If it is a list, all list members must be Fields living over the same
        one-dimensional `RGSpace` or `PowerSpace`.
    title: string
        title of the plot
    xlabel: string
        label for the x axis
    ylabel: string
        label for the y axis
    [xyz]min, [xyz]max: float
        limits for the values to plot
    colormap: string
        color map to use for the plot (if it is a 2D plot)
    linewidth: float or list of floats
        line width
    label: string of list of strings
        annotation string
    alpha: float or list of floats
        transparency value
    """
    _plots.append(f)
    _kwargs.append(kwargs)


def plot_finish(**kwargs):
    """Plot the accumulated list of figures.

    Parameters
    ----------
    title: string
        title of the full plot
    nx, ny: integer (default: square root of the numer of plots, rounded up)
        number of subplots to use in x- and y-direction
    xsize, ysize: float (default: 6)
        dimensions of the full plot in inches
    name: string (default: "")
        if left empty, the plot will be shown on the screen,
        otherwise it will be written to a file with the given name.
        Supported extensions: .png and .pdf
    """
    global _plots, _kwargs
    import matplotlib.pyplot as plt
    nplot = len(_plots)
    fig = plt.figure()
    if "title" in kwargs:
        plt.suptitle(kwargs.pop("title"))
    nx = kwargs.pop("nx", int(np.ceil(np.sqrt(nplot))))
    ny = kwargs.pop("ny", int(np.ceil(np.sqrt(nplot))))
    if nx*ny < nplot:
        raise ValueError(
            'Figure dimensions not sufficient for number of plots. '
            'Available plot slots: {}, number of plots: {}'
            .format(nx*ny, nplot))
    xsize = kwargs.pop("xsize", 6)
    ysize = kwargs.pop("ysize", 6)
    fig.set_size_inches(xsize, ysize)
    for i in range(nplot):
        ax = fig.add_subplot(ny, nx, i+1)
        _plot(_plots[i], ax, **_kwargs[i])
    fig.tight_layout()
    _makeplot(kwargs.pop("name", None))
    _plots = []
    _kwargs = []
