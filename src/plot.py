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

import os
from datetime import datetime as dt
from itertools import product
from warnings import warn

import numpy as np

from .domain_tuple import DomainTuple
from .domains.gl_space import GLSpace
from .domains.hp_space import HPSpace
from .domains.power_space import PowerSpace
from .domains.rg_space import RGSpace
from .field import Field
from .minimization.iteration_controllers import EnergyHistory
from .multi_field import MultiField
from .utilities import check_object_identity, myassert

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


def _rgb_data(spectral_cube):
    _xyz = np.array(
          [[0.000160, 0.000662, 0.002362, 0.007242, 0.019110,
            0.043400, 0.084736, 0.140638, 0.204492, 0.264737,
            0.314679, 0.357719, 0.383734, 0.386726, 0.370702,
            0.342957, 0.302273, 0.254085, 0.195618, 0.132349,
            0.080507, 0.041072, 0.016172, 0.005132, 0.003816,
            0.015444, 0.037465, 0.071358, 0.117749, 0.172953,
            0.236491, 0.304213, 0.376772, 0.451584, 0.529826,
            0.616053, 0.705224, 0.793832, 0.878655, 0.951162,
            1.014160, 1.074300, 1.118520, 1.134300, 1.123990,
            1.089100, 1.030480, 0.950740, 0.856297, 0.754930,
            0.647467, 0.535110, 0.431567, 0.343690, 0.268329,
            0.204300, 0.152568, 0.112210, 0.081261, 0.057930,
            0.040851, 0.028623, 0.019941, 0.013842, 0.009577,
            0.006605, 0.004553, 0.003145, 0.002175, 0.001506,
            0.001045, 0.000727, 0.000508, 0.000356, 0.000251,
            0.000178, 0.000126, 0.000090, 0.000065, 0.000046,
            0.000033],
           [0.000017, 0.000072, 0.000253, 0.000769, 0.002004,
            0.004509, 0.008756, 0.014456, 0.021391, 0.029497,
            0.038676, 0.049602, 0.062077, 0.074704, 0.089456,
            0.106256, 0.128201, 0.152761, 0.185190, 0.219940,
            0.253589, 0.297665, 0.339133, 0.395379, 0.460777,
            0.531360, 0.606741, 0.685660, 0.761757, 0.823330,
            0.875211, 0.923810, 0.961988, 0.982200, 0.991761,
            0.999110, 0.997340, 0.982380, 0.955552, 0.915175,
            0.868934, 0.825623, 0.777405, 0.720353, 0.658341,
            0.593878, 0.527963, 0.461834, 0.398057, 0.339554,
            0.283493, 0.228254, 0.179828, 0.140211, 0.107633,
            0.081187, 0.060281, 0.044096, 0.031800, 0.022602,
            0.015905, 0.011130, 0.007749, 0.005375, 0.003718,
            0.002565, 0.001768, 0.001222, 0.000846, 0.000586,
            0.000407, 0.000284, 0.000199, 0.000140, 0.000098,
            0.000070, 0.000050, 0.000036, 0.000025, 0.000018,
            0.000013],
           [0.000705, 0.002928, 0.010482, 0.032344, 0.086011,
            0.197120, 0.389366, 0.656760, 0.972542, 1.282500,
            1.553480, 1.798500, 1.967280, 2.027300, 1.994800,
            1.900700, 1.745370, 1.554900, 1.317560, 1.030200,
            0.772125, 0.570060, 0.415254, 0.302356, 0.218502,
            0.159249, 0.112044, 0.082248, 0.060709, 0.043050,
            0.030451, 0.020584, 0.013676, 0.007918, 0.003988,
            0.001091, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000]])

    MATRIX_SRGB_D65 = np.array(
            [[3.2404542, -1.5371385, -0.4985314],
             [-0.9692660,  1.8760108,  0.0415560],
             [0.0556434, -0.2040259,  1.0572252]])

    def _gammacorr(inp):
        mask = np.zeros(inp.shape, dtype=np.float64)
        mask[inp <= 0.0031308] = 1.
        r1 = 12.92*inp
        a = 0.055
        r2 = (1 + a) * (np.maximum(inp, 0.0031308) ** (1/2.4)) - a
        return r1*mask + r2*(1.-mask)

    def lambda2xyz(lam):
        lammin = 380.
        lammax = 780.
        lam = np.asarray(lam, dtype=np.float64)
        lam = np.clip(lam, lammin, lammax)

        idx = (lam-lammin)/(lammax-lammin)*(_xyz.shape[1]-1)
        ii = np.maximum(0, np.minimum(79, int(idx)))
        w1 = 1.-(idx-ii)
        w2 = 1.-w1
        c = w1*_xyz[:, ii] + w2*_xyz[:, ii+1]
        return c

    def getxyz(n):
        E0, E1 = 1./700., 1./400.
        E = E0 + np.arange(n)*(E1-E0)/(n-1)
        res = np.zeros((3, n), dtype=np.float64)
        for i in range(n):
            res[:, i] = lambda2xyz(1./E[i])
        return res

    def to_logscale(arr, lo, hi):
        res = arr.clip(lo, hi)
        res = np.log(res/hi)
        tmp = np.log(hi/lo)
        res += tmp
        res /= tmp
        return res

    shp = spectral_cube.shape[:-1]+(3,)
    spectral_cube = spectral_cube.reshape((-1, spectral_cube.shape[-1]))
    xyz = getxyz(spectral_cube.shape[-1])
    xyz_data = np.tensordot(spectral_cube, xyz, axes=[-1, -1])
    xyz_data /= xyz_data.max()
    xyz_data = to_logscale(xyz_data, max(1e-3, xyz_data.min()), 1.)
    rgb_data = xyz_data.copy()
    for x in range(xyz_data.shape[0]):
        rgb_data[x] = _gammacorr(np.matmul(MATRIX_SRGB_D65, xyz_data[x]))
    rgb_data = rgb_data.clip(0., 1.)
    return rgb_data.reshape(shp)


def _find_closest(A, target):
    # A must be sorted
    idx = np.clip(A.searchsorted(target), 1, len(A)-1)
    idx -= target - A[idx-1] < A[idx] - target
    return idx


def _makeplot(name, block=True, dpi=None):
    import matplotlib.pyplot as plt

    if name is None:
        plt.show(block=block)
        if block:
            plt.close()
        return
    extension = os.path.splitext(name)[1]
    if extension in (".pdf", ".png", ".svg"):
        args = {}
        if dpi is not None:
            args['dpi'] = float(dpi)
        plt.savefig(name, **args)
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
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap

    try:
        if _register_cmaps._cmaps_registered:
            return
    except AttributeError:
        _register_cmaps._cmaps_registered = True

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

    mpl.colormaps.register(cmap=LinearSegmentedColormap("Planck-like",
                                                        planckcmap))
    mpl.colormaps.register(cmap=LinearSegmentedColormap("High Energy",
                                                        he_cmap))
    mpl.colormaps.register(cmap=LinearSegmentedColormap("Faraday Map",
                                                        fd_cmap))
    mpl.colormaps.register(cmap=LinearSegmentedColormap("Faraday Uncertainty",
                                                        fdu_cmap))
    mpl.colormaps.register(cmap=LinearSegmentedColormap("Plus Minus", pm_cmap))


def _extract_list_kwargs(kwargs, keys, n):
    tmp = {}
    for kk in keys:
        val = kwargs.pop(kk, None)
        tmp[kk] = val if isinstance(val, list) else n*[val]
    with_legend = "label" in keys and any(ll is not None for ll in tmp["label"])
    return [{kk: vv[i] for kk, vv in tmp.items()} for i in range(n)], with_legend


def _plot_history(f, ax, **kwargs):
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter, date2num

    for i, fld in enumerate(f):
        if not isinstance(fld, EnergyHistory):
            raise TypeError
    label = kwargs.pop("label", None)
    if not isinstance(label, list):
        label = [label] * len(f)
    alpha = kwargs.pop("alpha", None)
    if not isinstance(alpha, list):
        alpha = [alpha] * len(f)
    color = kwargs.pop("color", None)
    if not isinstance(color, list):
        color = [color] * len(f)
    size = kwargs.pop("s", None)
    if not isinstance(size, list):
        size = [size] * len(f)
    ax.set_title(kwargs.pop("title", ""))
    ax.set_xlabel(kwargs.pop("xlabel", ""))
    ax.set_ylabel(kwargs.pop("ylabel", ""))

    skip_timestamp_conversion = kwargs.pop("skip_timestamp_conversion", False)
    energy_differences = kwargs.pop("plot_energy_differences", False)

    plt.xscale(kwargs.pop("xscale", "linear"))
    default_yscale = 'linear' if not energy_differences else 'log'
    plt.yscale(kwargs.pop("yscale", default_yscale))

    mi, ma = np.inf, -np.inf

    for i, fld in enumerate(f):
        kwargs = {'alpha': alpha[i], 's': size[i], 'color': color[i]}

        if skip_timestamp_conversion:
            xcoord = fld.time_stamps
        else:
            xcoord = date2num([dt.fromtimestamp(ts) for ts in fld.time_stamps])

        if not energy_differences:
            ycoord = fld.energy_values
            ax.scatter(xcoord, ycoord, label=label[i], **kwargs)
        else:
            E = np.array(fld.energy_values)
            dE = E[1:] - E[:-1]
            xcoord = np.array(xcoord[1:])
            idx_pos = (dE > 0)
            idx_neg = (dE < 0)
            label_pos = label[i] + ' (pos)' if label[i] is not None else None
            label_neg = label[i] + ' (neg)' if label[i] is not None else None
            ax.scatter(xcoord[idx_pos], dE[idx_pos], marker='^',
                       label=label_pos, **kwargs)
            ax.scatter(xcoord[idx_neg], dE[idx_neg], marker='v',
                       label=label_neg, **kwargs)

        mi, ma = min([min(xcoord), mi]), max([max(xcoord), ma])

    delta = (ma-mi)*0.05
    if delta == 0.:
        delta = 1.
    ax.set_xlim((mi-delta, ma+delta))
    if not skip_timestamp_conversion:
        xfmt = DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(xfmt)
    _limit_xy(**kwargs)
    if label != ([None]*len(f)):
        plt.legend(loc="upper right")


def _plot1D(f, ax, **kwargs):
    import matplotlib.pyplot as plt

    dom = f[0].domain[0]
    add_kwargs, with_legend = _extract_list_kwargs(kwargs, ("label", "alpha", "color", "linewidth"), len(f))

    if isinstance(dom, RGSpace):
        plt.yscale(kwargs.pop("yscale", "linear"))
        npoints = dom.shape[0]
        dist = dom.distances[0]
        xcoord = np.arange(npoints, dtype=np.float64)*dist
        for i, fld in enumerate(f):
            ycoord = fld.val
            plt.plot(xcoord, ycoord, **add_kwargs[i])
    elif isinstance(dom, PowerSpace):
        plt.xscale(kwargs.pop("xscale", "log"))
        plt.yscale(kwargs.pop("yscale", "log"))
        xcoord = dom.k_lengths
        for i, fld in enumerate(f):
            ycoord = fld.val_rw()
            ycoord[0] = ycoord[1]
            plt.plot(xcoord, ycoord, **add_kwargs[i])
    else:
        raise RuntimeError("This point should never be reached")

    _limit_xy(**kwargs)
    if with_legend:
        ax.legend(loc="upper right")


def plottable1D(f):
    dom = f[0].domain
    is_1d_plottable = isinstance(dom[0], (RGSpace, PowerSpace))
    is_1d_plottable &= (len(dom) == 1) and (len(dom.shape) == 1)
    is_1d_plottable &= all(dom == el.domain for el in f)
    return is_1d_plottable


def plottable2D(fld, f_space=1):
    dom = fld.domain
    if not isinstance(dom, DomainTuple) or len(dom) > 2:
        return False
    if f_space not in [0, 1]:
        return False
    x_space = 0
    if len(dom) == 2:
        x_space = 1 - f_space
        if (not isinstance(dom[f_space], RGSpace)) or len(dom[f_space].shape) != 1:
            return False
    elif len(dom) == 0:
        return False
    if not isinstance(dom[x_space], (RGSpace, HPSpace, GLSpace)):
        return False
    if isinstance(dom[x_space], RGSpace) and not len(dom[x_space].shape) == 2:
        return False
    return True


def _plotting_args_2D(fld, f_space=1):
    from .sugar import makeField

    # check for multifrequency plotting
    have_rgb, rgb = False, None
    x_space = 0
    dom = fld.domain
    if len(dom) == 1:
        x_space = 0
    elif len(dom) == 2:
        x_space = 1 - f_space
        # Only one frequency?
        if dom[f_space].shape[0] == 1:
            fld = makeField(fld.domain[x_space], fld.val.squeeze(axis=dom.axes[f_space]))
        else:
            val = fld.val
            if f_space == 0:
                val = np.moveaxis(val, 0, -1)
            rgb = _rgb_data(val)
            have_rgb = True
    else:  # "DomainTuple can only have one or two entries.
        raise ValueError('check plottable2D before using this function')
    return fld, x_space, have_rgb, rgb


def _plot2D(f, ax, **kwargs):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    f = f[0]
    dom = f.domain

    f, x_space, have_rgb, rgb = _plotting_args_2D(f, kwargs.pop("freq_space_idx", 1))

    foo = kwargs.pop("norm", None)
    norm = {} if foo is None else {'norm': foo}

    foo = kwargs.pop("aspect", None)
    aspect = {} if foo is None else {'aspect': foo}

    dom = dom[x_space]
    if not have_rgb:
        cmap = kwargs.pop("cmap", plt.rcParams['image.cmap'])

    if isinstance(dom, RGSpace):
        nx, ny = dom.shape
        dx, dy = dom.distances
        if have_rgb:
            im = ax.imshow(
                rgb, extent=[0, nx*dx, 0, ny*dy], origin="lower", **norm,
                **aspect)
        else:
            im = ax.imshow(
                f.val.T, extent=[0, nx*dx, 0, ny*dy],
                vmin=kwargs.get("vmin"), vmax=kwargs.get("vmax"),
                cmap=cmap, origin="lower", **norm, **aspect)
            cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
        _limit_xy(**kwargs)
        return
    elif isinstance(dom, (HPSpace, GLSpace)):
        from ducc0.healpix import Healpix_Base
        from ducc0.misc import GL_thetas

        xsize = 800
        res, mask, theta, phi = _mollweide_helper(xsize)
        if have_rgb:
            res = np.full(shape=res.shape+(3,), fill_value=1.,
                          dtype=np.float64)

        if isinstance(dom, HPSpace):
            ptg = np.empty((phi.size, 2), dtype=np.float64)
            ptg[:, 0] = theta
            ptg[:, 1] = phi
            base = Healpix_Base(int(np.sqrt(dom.size//12)), "RING")
            if have_rgb:
                res[mask] = rgb[base.ang2pix(ptg)]
            else:
                res[mask] = f.val[base.ang2pix(ptg)]
        else:
            ra = np.linspace(0, 2*np.pi, dom.nlon+1)
            dec = GL_thetas(dom.nlat)
            ilat = _find_closest(dec, theta)
            ilon = _find_closest(ra, phi)
            ilon = np.where(ilon == dom.nlon, 0, ilon)
            if have_rgb:
                res[mask] = rgb[ilat*dom[0].nlon + ilon]
            else:
                res[mask] = f.val[ilat*dom.nlon + ilon]
        plt.axis('off')
        if have_rgb:
            plt.imshow(res, origin="lower")
        else:
            plt.imshow(res, vmin=kwargs.get("vmin"), vmax=kwargs.get("vmax"),
                       norm=norm.get('norm'), cmap=cmap, origin="lower")
            plt.colorbar(orientation="horizontal")
        return
    raise ValueError("Field type not(yet) supported")


def _plotHist(f, ax, **kwargs):
    add_kwargs, with_legend = _extract_list_kwargs(kwargs, ("label", "alpha", "color", "range"),
                                                   len(f))
    add_kwargs2 = {
        "log": kwargs.pop("log", False),
        "density": kwargs.pop("density", False),
        "bins": kwargs.pop("bins", 50)
    }
    for i, fld in enumerate(f):
        ax.hist(fld.val.ravel(), **add_kwargs[i], **add_kwargs2)
    if with_legend:
        ax.legend(loc="upper right")


def _plot(f, ax, **kwargs):
    _register_cmaps()
    if isinstance(f[0], EnergyHistory):
        _plot_history(f, ax, **kwargs)
        return

    ax.set_title(kwargs.pop("title", ""))
    ax.set_xlabel(kwargs.pop("xlabel", ""))
    ax.set_ylabel(kwargs.pop("ylabel", ""))

    if plottable1D(f):
        _plot1D(f, ax, **kwargs)
        return
    if plottable2D(f[0], kwargs.get("freq_space_idx", 1)):
        _plot2D(f, ax, **kwargs)
        return
    _plotHist(f, ax, **kwargs)


class Plot:
    def __init__(self):
        self._plots = []
        self._kwargs = []

    def add(self, f, **kwargs):
        """Add a figure to the current list of plots.

        Notes
        -----
        After doing one or more calls `add()`, one needs to call `output()` to
        show or save the plot.

        Parameters
        ----------
        f : :class:`nifty8.field.Field` or list of :class:`nifty8.field.Field` or None
            If `f` is a single Field, it must be defined on a single `RGSpace`,
            `PowerSpace`, `HPSpace`, `GLSpace`.
            If it is a list, all list members must be Fields defined over the
            same one-dimensional `RGSpace` or `PowerSpace`.
            If `f` is `None`, an empty panel will be displayed.

        Optional Parameters
        -------------------
        title: string
            Title of the plot.
        xlabel: string
            Label for the x axis.
        ylabel: string
            Label for the y axis.
        [xyz]min, [xyz]max: float
            Limits for the values to plot.
        cmap: string
            Color map to use for the plot (if it is a 2D plot).
        linewidth: float or list of floats
            Line width.
        label: string of list of strings
            Annotation string.
        alpha: float or list of floats
            Transparency value.
        freq_space_idx: int
            for multi-frequency plotting: index of frequency space in domain
        """
        if f is None:
            self._plots.append(None)
            self._kwargs.append({})
            return
        if isinstance(f, (MultiField, Field, EnergyHistory)):
            f = [f]
        if not isinstance(f[0], (MultiField, Field, EnergyHistory)):
            raise TypeError("Incorrect data type. You can only add Fields or EnergyHistories, or None")
        if hasattr(f, "__len__") and all(isinstance(ff, MultiField) for ff in f):
            for kk in f[0].domain.keys():
                self._plots.append([ff[kk] for ff in f])
                mykwargs = kwargs.copy()
                if 'title' in kwargs:
                    mykwargs['title'] = "{} {}".format(kk, kwargs['title'])
                else:
                    mykwargs['title'] = "{}".format(kk)
                self._kwargs.append(mykwargs)
            return

        if isinstance(f[0], EnergyHistory):
            dom = None
        else:
            dom = f[0].domain

        if isinstance(dom, DomainTuple) \
                and any(isinstance(dd, RGSpace) for dd in dom) \
                and not "freq_space_idx" in kwargs \
                and not plottable2D(f[0], kwargs.get("freq_space_idx", 1)):
            from .sugar import makeField

            dims = [len(dd.shape) for dd in dom]
            # One space is 2d, the rest is 1d
            if np.sum(np.array(dims) == 2) == 1 and np.sum(np.array(dims) == 1) == len(dims) - 1:
                twod_index = dims.index(2)
                sizes = [dd.size for dd in dom]
                del(sizes[twod_index])
                for multi_index in product(*[tuple(range(ii)) for ii in sizes]):
                    multi_index = list(multi_index)
                    multi_index.insert(twod_index, slice(None))
                    multi_index.insert(twod_index, slice(None))
                    for ifield in range(len(f)):
                        arr = f[ifield].val[tuple(multi_index)]
                        myassert(arr.ndim == 2)
                        self._plots.append([makeField(dom[twod_index], arr)])
                        self._kwargs.append(kwargs)
                return
        self._plots.append(f)
        self._kwargs.append(kwargs)

    def output(self, **kwargs):
        """Plot the accumulated list of figures.

        Parameters
        ----------
        title: string
            Title of the full plot.
        nx, ny: int
            Number of subplots to use in x- and y-direction.
            Default: square root of the numer of plots, rounded up.
        xsize, ysize: float
            Dimensions of the full plot in inches. Default: 6.
        name: string
            If left empty, the plot will be shown on the screen,
            otherwise it will be written to a file with the given name.
            Supported extensions: .png and .pdf. Default: None.
        block: bool
            Override the blocking behavior of the non-interactive plotting
            mode. The plot will not be closed in this case but is left open!
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warn("Since matplotlib is not installed, NIFTy will not generate any plots.")
            return

        nplot = len(self._plots)
        if nplot == 0:
            raise ValueError("Use .add to add plots to your plotting routine.")
        fig = plt.figure()
        if "title" in kwargs:
            plt.suptitle(kwargs.pop("title"))
        nx = kwargs.pop("nx", 0)
        ny = kwargs.pop("ny", 0)
        if nx == ny == 0:
            ny = int(np.ceil(np.sqrt(nplot)))
            nx = int(np.ceil(nplot/ny))
            myassert(nx*ny >= nplot)
        elif nx == 0:
            nx = int(np.ceil(nplot/ny))
        elif ny == 0:
            ny = int(np.ceil(nplot/nx))
        if nx*ny < nplot:
            raise ValueError(
                'Figure dimensions not sufficient for number of plots. '
                'Available plot slots: {}, number of plots: {}'
                .format(nx*ny, nplot))
        xsize = kwargs.pop("xsize", 6*nx)
        ysize = kwargs.pop("ysize", 6*ny)
        fig.set_size_inches(xsize, ysize)
        for i in range(nplot):
            if self._plots[i] is None:
                continue
            ax = fig.add_subplot(ny, nx, i+1)
            _plot(self._plots[i], ax, **self._kwargs[i])
        fig.tight_layout()
        _makeplot(kwargs.pop("name", None),
                  block=kwargs.pop("block", True),
                  dpi=kwargs.pop("dpi", None))
