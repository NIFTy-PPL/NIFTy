from __future__ import division
import matplotlib.pyplot as plt
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
    if name is None:
        plt.show()
        return
    extension = os.path.splitext(name)[1]
    if extension==".pdf":
        plt.savefig(name)
        plt.close()
    elif extension==".png":
        plt.savefig(name)
        plt.close()
    #elif extension==".html":
        #import mpld3
        #mpld3.save_html(plt.gcf(),fileobj=name,no_extras=True)
        #import plotly.offline as py
        #import plotly.tools as tls
        #plotly_fig = tls.mpl_to_plotly(plt.gcf())
        #py.plot(plotly_fig,filename=name)
        #py.plot_mpl(plt.gcf(),filename=name)
        #import bokeh
        #bokeh.mpl.to_bokeh(plt.gcf())
    else:
        raise ValueError("file format not understood")

def plot (f,name=None):
    if not isinstance(f,Field):
        raise TypeError("incorrect data type")
    if len(f.domain)!=1:
        raise ValueError("input field must have exactly one domain")

    dom = f.domain[0]
    plt.gcf().set_size_inches(12,12)

    if isinstance(dom, RGSpace):
        if len(dom.shape)==1:
            npoints = dom.shape[0]
            dist = dom.distances[0]
            xcoord = np.arange(npoints,dtype=np.float64)*dist
            ycoord = f.val
            plt.plot(xcoord, ycoord)
            _makeplot(name)
            return
        elif len(dom.shape)==2:
            nx = dom.shape[0]
            ny = dom.shape[1]
            dx = dom.distances[0]
            dy = dom.distances[1]
            xc = np.arange(nx,dtype=np.float64)*dx
            yc = np.arange(ny,dtype=np.float64)*dy
            plt.imshow(f.val,extent=[xc[0],xc[-1],yc[0],yc[-1]])
            plt.colorbar()
            _makeplot(name)
            return
    elif isinstance(dom, PowerSpace):
        xcoord = dom.kindex
        ycoord = f.val
        plt.xscale('log')
        plt.yscale('log')
        plt.title('power')
        plt.plot(xcoord, ycoord)
        _makeplot(name)
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
        plt.imshow(res)
        plt.colorbar()
        _makeplot(name)
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
        plt.imshow(res)
        plt.colorbar()
        _makeplot(name)
        return

    raise ValueError("Field type not(yet) supported")
