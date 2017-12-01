from __future__ import division
import numpy as np
from ..import Field, RGSpace, HPSpace, GLSpace, PowerSpace, dobj
import os

# relevant properties:
# - x/y size
# - x/y/z log
# - x/y/z min/max
# - colorbar/colormap
# - axis on/off
# - title
# - axis labels


def _mollweide_helper(xsize):
    xsize = int(xsize)
    ysize = xsize//2
    res = np.full(shape=(ysize, xsize), fill_value=np.nan, dtype=np.float64)
    xc = (xsize-1)*0.5
    yc = (ysize-1)*0.5
    u, v = np.meshgrid(np.arange(xsize), np.arange(ysize))
    u = 2*(u-xc)/(xc/1.02)
    v = (v-yc)/(yc/1.02)

    mask = np.where((u*u*0.25 + v*v) <= 1.)
    t1 = v[mask]
    theta = 0.5*np.pi-(
        np.arcsin(2/np.pi*(np.arcsin(t1) + t1*np.sqrt((1.-t1)*(1+t1)))))
    phi = -0.5*np.pi*u[mask]/np.maximum(np.sqrt((1-t1)*(1+t1)), 1e-6)
    phi = np.where(phi < 0, phi+2*np.pi, phi)
    return res, mask, theta, phi


def _find_closest(A, target):
    # A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
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
    if extension == ".pdf":
        plt.savefig(name)
        plt.close()
    elif extension == ".png":
        plt.savefig(name)
        plt.close()
    # elif extension==".html":
        # import mpld3
        # mpld3.save_html(plt.gcf(),fileobj=name,no_extras=True)
        # import plotly.offline as py
        # import plotly.tools as tls
        # plotly_fig = tls.mpl_to_plotly(plt.gcf())
        # py.plot(plotly_fig,filename=name)
        # py.plot_mpl(plt.gcf(),filename=name)
        # import bokeh
        # bokeh.mpl.to_bokeh(plt.gcf())
    else:
        raise ValueError("file format not understood")


def _limit_xy(**kwargs):
    import matplotlib.pyplot as plt
    x1, x2, y1, y2 = plt.axis()
    x1 = _get_kw("xmin", x1, **kwargs)
    x2 = _get_kw("xmax", x2, **kwargs)
    y1 = _get_kw("ymin", y1, **kwargs)
    y2 = _get_kw("xmax", y2, **kwargs)
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


def _get_kw(kwname, kwdefault=None, **kwargs):
    if kwargs.get(kwname) is not None:
        return kwargs.get(kwname)
    return kwdefault


def plot(f, **kwargs):
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
                    (isinstance(dom[0], RGSpace) and len(dom[0].shape)==1)):
                raise ValueError("PowerSpace or 1D RGSpace required")

    dom = dom[0]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    xsize = _get_kw("xsize", 6, **kwargs)
    ysize = _get_kw("ysize", 6, **kwargs)
    fig.set_size_inches(xsize, ysize)
    ax.set_title(_get_kw("title", "", **kwargs))
    ax.set_xlabel(_get_kw("xlabel", "", **kwargs))
    ax.set_ylabel(_get_kw("ylabel", "", **kwargs))
    cmap = _get_kw("colormap", plt.rcParams['image.cmap'], **kwargs)
    if isinstance(dom, RGSpace):
        if len(dom.shape) == 1:
            npoints = dom.shape[0]
            dist = dom.distances[0]
            xcoord = np.arange(npoints, dtype=np.float64)*dist
            for fld in f:
                ycoord = dobj.to_global_data(fld.val)
                plt.plot(xcoord, ycoord)
            _limit_xy(**kwargs)
            _makeplot(kwargs.get("name"))
            return
        elif len(dom.shape) == 2:
            f = f[0]
            nx = dom.shape[0]
            ny = dom.shape[1]
            dx = dom.distances[0]
            dy = dom.distances[1]
            xc = np.arange(nx, dtype=np.float64)*dx
            yc = np.arange(ny, dtype=np.float64)*dy
            im = ax.imshow(dobj.to_global_data(f.val),
                           extent=[xc[0], xc[-1], yc[0], yc[-1]],
                           vmin=kwargs.get("zmin"),
                           vmax=kwargs.get("zmax"), cmap=cmap, origin="lower")
            # from mpl_toolkits.axes_grid1 import make_axes_locatable
            # divider = make_axes_locatable(ax)
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            # plt.colorbar(im,cax=cax)
            plt.colorbar(im)
            _limit_xy(**kwargs)
            _makeplot(kwargs.get("name"))
            return
    elif isinstance(dom, PowerSpace):
        plt.xscale('log')
        plt.yscale('log')
        plt.title('power')
        xcoord = dom.k_lengths
        for fld in f:
            ycoord = dobj.to_global_data(fld.val)
            plt.plot(xcoord, ycoord)
        _limit_xy(**kwargs)
        _makeplot(kwargs.get("name"))
        return
    elif isinstance(dom, HPSpace):
        f = f[0]
        import pyHealpix
        xsize = 800
        res, mask, theta, phi = _mollweide_helper(xsize)

        ptg = np.empty((phi.size, 2), dtype=np.float64)
        ptg[:, 0] = theta
        ptg[:, 1] = phi
        base = pyHealpix.Healpix_Base(int(np.sqrt(f.val.size//12)), "RING")
        res[mask] = dobj.to_global_data(f.val)[base.ang2pix(ptg)]
        plt.axis('off')
        plt.imshow(res, vmin=kwargs.get("zmin"), vmax=kwargs.get("zmax"),
                   cmap=cmap, origin="lower")
        plt.colorbar(orientation="horizontal")
        _makeplot(kwargs.get("name"))
        return
    elif isinstance(dom, GLSpace):
        f = f[0]
        import pyHealpix
        xsize = 800
        res, mask, theta, phi = _mollweide_helper(xsize)
        ra = np.linspace(0, 2*np.pi, dom.nlon+1)
        dec = pyHealpix.GL_thetas(dom.nlat)
        ilat = _find_closest(dec, theta)
        ilon = _find_closest(ra, phi)
        ilon = np.where(ilon == dom.nlon, 0, ilon)
        res[mask] = dobj.to_global_data(f.val)[ilat*dom.nlon + ilon]

        plt.axis('off')
        plt.imshow(res, vmin=kwargs.get("zmin"), vmax=kwargs.get("zmax"),
                   cmap=cmap, origin="lower")
        plt.colorbar(orientation="horizontal")
        _makeplot(kwargs.get("name"))
        return

    raise ValueError("Field type not(yet) supported")
