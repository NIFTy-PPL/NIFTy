from __future__ import division
import numpy as np
from ..import Field, RGSpace, HPSpace, GLSpace, PowerSpace
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
    res = np.full(shape=(ysize, xsize), fill_value=np.nan,
                  dtype=np.float64)
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
    if name is None:
        plt.show()
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
    if (kwargs.get("xmin")) is not None:
        x1 = kwargs.get("xmin")
    if (kwargs.get("xmax")) is not None:
        x2 = kwargs.get("xmax")
    if (kwargs.get("ymin")) is not None:
        y1 = kwargs.get("ymin")
    if (kwargs.get("ymax")) is not None:
        y2 = kwargs.get("ymax")
    plt.axis((x1, x2, y1, y2))


def _register_cmaps():
    try:
        if _register_cmaps._cmaps_registered:
            return
    except AttributeError:
        _register_cmaps._cmaps_registered = True

    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.pyplot as plt
    planckcmap = {'red':   ((0.0, 0.0, 0.0),
                            (0.4, 0.0, 0.0),
                            (0.5, 1.0, 1.0),
                            (0.7, 1.0, 1.0),
                            (0.8, 0.83, 0.83),
                            (0.9, 0.67, 0.67),
                            (1.0, 0.5, 0.5)),
                  'green': ((0.0, 0.0, 0.0),
                            (0.2, 0.0, 0.0),
                            (0.3, 0.3, 0.3),
                            (0.4, 0.7, 0.7),
                            (0.5, 1.0, 1.0),
                            (0.6, 0.7, 0.7),
                            (0.7, 0.3, 0.3),
                            (0.8, 0.0, 0.0),
                            (1.0, 0.0, 0.0)),
                  'blue':  ((0.0, 0.5, 0.5),
                            (0.1, 0.67, 0.67),
                            (0.2, 0.83, 0.83),
                            (0.3, 1.0, 1.0),
                            (0.5, 1.0, 1.0),
                            (0.6, 0.0, 0.0),
                            (1.0, 0.0, 0.0))}
    he_cmap = {'red':   ((0.0, 0.0, 0.0),
                         (0.167, 0.0, 0.0),
                         (0.333, 0.5, 0.5),
                         (0.5, 1.0, 1.0),
                         (1.0, 1.0, 1.0)),
               'green': ((0.0, 0.0, 0.0),
                         (0.5, 0.0, 0.0),
                         (0.667, 0.5, 0.5),
                         (0.833, 1.0, 1.0),
                         (1.0, 1.0, 1.0)),
               'blue':  ((0.0, 0.0, 0.0),
                         (0.167, 1.0, 1.0),
                         (0.333, 0.5, 0.5),
                         (0.5, 0.0, 0.0),
                         (1.0, 1.0, 1.0))}
    fd_cmap = {'red':   ((0.0, 0.35, 0.35),
                         (0.1, 0.4, 0.4),
                         (0.2, 0.25, 0.25),
                         (0.41, 0.47, 0.47),
                         (0.5, 0.8, 0.8),
                         (0.56, 0.96, 0.96),
                         (0.59, 1.0, 1.0),
                         (0.74, 0.8, 0.8),
                         (0.8, 0.8, 0.8),
                         (0.9, 0.5, 0.5),
                         (1.0, 0.4, 0.4)),
               'green': ((0.0, 0.0, 0.0),
                         (0.2, 0.0, 0.0),
                         (0.362, 0.88, 0.88),
                         (0.5, 1.0, 1.0),
                         (0.638, 0.88, 0.88),
                         (0.8, 0.25, 0.25),
                         (0.9, 0.3, 0.3),
                         (1.0, 0.2, 0.2)),
               'blue':  ((0.0, 0.35, 0.35),
                         (0.1, 0.4, 0.4),
                         (0.2, 0.8, 0.8),
                         (0.26, 0.8, 0.8),
                         (0.41, 1.0, 1.0),
                         (0.44, 0.96, 0.96),
                         (0.5, 0.8, 0.8),
                         (0.59, 0.47, 0.47),
                         (0.8, 0.0, 0.0),
                         (1.0, 0.0, 0.0))}
    fdu_cmap = {'red':   ((0.0, 1.0, 1.0),
                          (0.1, 0.8, 0.8),
                          (0.2, 0.65, 0.65),
                          (0.41, 0.6, 0.6),
                          (0.5, 0.7, 0.7),
                          (0.56, 0.96, 0.96),
                          (0.59, 1.0, 1.0),
                          (0.74, 0.8, 0.8),
                          (0.8, 0.8, 0.8),
                          (0.9, 0.5, 0.5),
                          (1.0, 0.4, 0.4)),
                'green': ((0.0, 0.9, 0.9),
                          (0.2, 0.65, 0.65),
                          (0.362, 0.95, 0.95),
                          (0.5, 1.0, 1.0),
                          (0.638, 0.88, 0.88),
                          (0.8, 0.25, 0.25),
                          (0.9, 0.3, 0.3),
                          (1.0, 0.2, 0.2)),
                'blue':  ((0.0, 1.0, 1.0),
                          (0.1, 0.8, 0.8),
                          (0.2, 1.0, 1.0),
                          (0.41, 1.0, 1.0),
                          (0.44, 0.96, 0.96),
                          (0.5, 0.7, 0.7),
                          (0.59, 0.42, 0.42),
                          (0.8, 0.0, 0.0),
                          (1.0, 0.0, 0.0))}
    pm_cmap = {'red':   ((0.0, 1.0, 1.0),
                         (0.1, 0.96, 0.96),
                         (0.2, 0.84, 0.84),
                         (0.3, 0.64, 0.64),
                         (0.4, 0.36, 0.36),
                         (0.5, 0.0, 0.0),
                         (1.0, 0.0, 0.0)),
               'green': ((0.0, 0.5, 0.5),
                         (0.1, 0.32, 0.32),
                         (0.2, 0.18, 0.18),
                         (0.3, 0.08, 0.08),
                         (0.4, 0.02, 0.02),
                         (0.5, 0.0, 0.0),
                         (0.6, 0.02, 0.02),
                         (0.7, 0.08, 0.08),
                         (0.8, 0.18, 0.18),
                         (0.9, 0.32, 0.32),
                         (1.0, 0.5, 0.5)),
               'blue':  ((0.0, 0.0, 0.0),
                         (0.5, 0.0, 0.0),
                         (0.6, 0.36, 0.36),
                         (0.7, 0.64, 0.64),
                         (0.8, 0.84, 0.84),
                         (0.9, 0.96, 0.96),
                         (1.0, 1.0, 1.0))}

    plt.register_cmap(cmap=LinearSegmentedColormap("Planck-like", planckcmap))
    plt.register_cmap(cmap=LinearSegmentedColormap("High Energy", he_cmap))
    plt.register_cmap(cmap=LinearSegmentedColormap("Faraday Map", fd_cmap))
    plt.register_cmap(cmap=LinearSegmentedColormap("Faraday Uncertainty",
                                                   fdu_cmap))
    plt.register_cmap(cmap=LinearSegmentedColormap("Plus Minus", pm_cmap))


def plot(f, **kwargs):
    import matplotlib.pyplot as plt
    _register_cmaps()
    if not isinstance(f, Field):
        raise TypeError("incorrect data type")
    if len(f.domain) != 1:
        raise ValueError("input field must have exactly one domain")

    dom = f.domain[0]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    xsize, ysize = 6, 6
    if kwargs.get("xsize") is not None:
        xsize = kwargs.get("xsize")
    if kwargs.get("ysize") is not None:
        ysize = kwargs.get("ysize")
    fig.set_size_inches(xsize, ysize)

    if kwargs.get("title") is not None:
        ax.set_title(kwargs.get("title"))
    if kwargs.get("xlabel") is not None:
        ax.set_xlabel(kwargs.get("xlabel"))
    if kwargs.get("ylabel") is not None:
        ax.set_ylabel(kwargs.get("ylabel"))
    cmap = plt.rcParams['image.cmap']
    if kwargs.get("colormap") is not None:
        cmap = kwargs.get("colormap")
    if isinstance(dom, RGSpace):
        if len(dom.shape) == 1:
            npoints = dom.shape[0]
            dist = dom.distances[0]
            xcoord = np.arange(npoints, dtype=np.float64)*dist
            ycoord = f.val
            plt.plot(xcoord, ycoord)
            _limit_xy(**kwargs)
            _makeplot(kwargs.get("name"))
            return
        elif len(dom.shape) == 2:
            nx = dom.shape[0]
            ny = dom.shape[1]
            dx = dom.distances[0]
            dy = dom.distances[1]
            xc = np.arange(nx, dtype=np.float64)*dx
            yc = np.arange(ny, dtype=np.float64)*dy
            im = ax.imshow(f.val, extent=[xc[0], xc[-1], yc[0], yc[-1]],
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
        xcoord = dom.kindex
        ycoord = f.val
        plt.xscale('log')
        plt.yscale('log')
        plt.title('power')
        plt.plot(xcoord, ycoord)
        _limit_xy(**kwargs)
        _makeplot(kwargs.get("name"))
        return
    elif isinstance(dom, HPSpace):
        import pyHealpix
        xsize = 800
        res, mask, theta, phi = _mollweide_helper(xsize)

        ptg = np.empty((phi.size, 2), dtype=np.float64)
        ptg[:, 0] = theta
        ptg[:, 1] = phi
        base = pyHealpix.Healpix_Base(int(np.sqrt(f.val.size//12)), "RING")
        res[mask] = f.val[base.ang2pix(ptg)]
        plt.axis('off')
        plt.imshow(res, vmin=kwargs.get("zmin"), vmax=kwargs.get("zmax"),
                   cmap=cmap, origin="lower")
        plt.colorbar(orientation="horizontal")
        _makeplot(kwargs.get("name"))
        return
    elif isinstance(dom, GLSpace):
        import pyHealpix
        xsize = 800
        res, mask, theta, phi = _mollweide_helper(xsize)
        ra = np.linspace(0, 2*np.pi, dom.nlon+1)
        dec = pyHealpix.GL_thetas(dom.nlat)
        ilat = _find_closest(dec, theta)
        ilon = _find_closest(ra, phi)
        ilon = np.where(ilon == dom.nlon, 0, ilon)
        res[mask] = f.val[ilat*dom.nlon + ilon]

        plt.axis('off')
        plt.imshow(res, vmin=kwargs.get("zmin"), vmax=kwargs.get("zmax"),
                   cmap=cmap, origin="lower")
        plt.colorbar(orientation="horizontal")
        _makeplot(kwargs.get("name"))
        return

    raise ValueError("Field type not(yet) supported")
