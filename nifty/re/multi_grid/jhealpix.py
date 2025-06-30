# Re-implemantation of C based healpix reference: https://gitlab.mpcdf.mpg.de/mtr/healpix_reference

# /* -----------------------------------------------------------------------------
#  *
#  *  Copyright (C) 1997-2019 Krzysztof M. Gorski, Eric Hivon, Martin Reinecke,
#  *                          Benjamin D. Wandelt, Anthony J. Banday,
#  *                          Matthias Bartelmann,
#  *                          Reza Ansari & Kenneth M. Ganga
#  *
#  *  Implementation of the Healpix bare bones C library
#  *
#  *  Licensed under a 3-clause BSD style license - see LICENSE
#  *
#  *  For more information on HEALPix and additional software packages, see
#  *  https://healpix.sourceforge.io/
#  *
#  *  If you are using this code in your own packages, please consider citing
#  *  the original paper in your publications:
#  *  K.M. Gorski et al., 2005, Ap.J., 622, p.759
#  *  (http://adsabs.harvard.edu/abs/2005ApJ...622..759G)
#  *
#  *----------------------------------------------------------------------------*/

from functools import partial
import jax
import jax.experimental
import jax.experimental.checkify
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp

from dataclasses import dataclass
import numpy as np


PI = 3.141592653589793238462643383279502884197
JRLL = jnp.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
JPLL = jnp.array([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7])

NB_XOFFSET = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
NB_YOFFSET = np.array([0, 1, 1, 1, 0, -1, -1, -1])
NB_FACEARRAY = jnp.array(
    [
        [8, 9, 10, 11, -1, -1, -1, -1, 10, 11, 8, 9],  # S
        [5, 6, 7, 4, 8, 9, 10, 11, 9, 10, 11, 8],  # SE
        [-1, -1, -1, -1, 5, 6, 7, 4, -1, -1, -1, -1],  # E
        [4, 5, 6, 7, 11, 8, 9, 10, 11, 8, 9, 10],  # SW
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # center
        [1, 2, 3, 0, 0, 1, 2, 3, 5, 6, 7, 4],  # NE
        [-1, -1, -1, -1, 7, 4, 5, 6, -1, -1, -1, -1],  # W
        [3, 0, 1, 2, 3, 0, 1, 2, 4, 5, 6, 7],  # NW
        [2, 3, 0, 1, -1, -1, -1, -1, 0, 1, 2, 3],  # N
    ]
)
NB_SWAPARRAY = jnp.array(
    [
        [0, 0, 3],  # S
        [0, 0, 6],  # SE
        [0, 0, 0],  # E
        [0, 0, 5],  # SW
        [0, 0, 0],  # center
        [5, 0, 0],  # NE
        [0, 0, 0],  # W
        [6, 0, 0],  # NW
        [3, 0, 0],  # N
    ]
)


BAD_NBR_SLCT = np.array([0, 0, 2, 4, 4, 6, 6, 0])
BAD_NBR_SHFT = np.array([0, -1, 0, 1, 0, -2, 0, -1])

# Conversions between continuous coordinate systems


@register_pytree_node_class
@dataclass()
class T_loc:
    z: float
    s: float
    phi: float

    def tree_flatten(self):
        children = (self.z, self.s, self.phi)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


# A structure describing the continuous Healpix coordinate system.
# f takes values in [0;11], x and y lie in [0.0; 1.0].
@register_pytree_node_class
@dataclass()
class T_hpc:
    x: float
    y: float
    f: np.int32

    def tree_flatten(self):
        children = (self.x, self.y, self.f)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


# A structure describing a location on the sphere. Theta is the co-latitude
# in radians (0 at the North Pole, increasing to pi at the South Pole. Phi is
# the azimuth in radians.
@register_pytree_node_class
@dataclass()
class T_ang:
    theta: float
    phi: float

    def tree_flatten(self):
        children = (self.theta, self.phi)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


# A structure describing a 3-vector with coordinates x, y and z.
@register_pytree_node_class
@dataclass()
class T_vec:
    x: float
    y: float
    z: float

    def tree_flatten(self):
        children = (self.x, self.y, self.z)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def norm(self):
        return jnp.sqrt(self.x**2 + self.y**2 + self.z**2)


# A structure describing the discrete Healpix coordinate system.
# f takes values in [0;11], x and y lie in [0; nside[.
@register_pytree_node_class
@dataclass()
class T_hpd:
    x: np.int64
    y: np.int64
    f: np.int32

    def tree_flatten(self):
        children = (self.x, self.y, self.f)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def loc2hpc(loc: T_loc) -> T_hpc:
    def loc2hpc_mid(loc, tt):  # Equatorial region
        temp1 = 0.5 + tt  # [0.5; 4.5)
        temp2 = loc.z * 0.75  # [-0.5; +0.5]
        jp = temp1 - temp2  # index of ascending edge line [0; 5)
        jm = temp1 + temp2  # index of descending edge line [0; 5)
        ifp = jp.astype(np.int64)  # int(jp)  # in {0, 4}
        ifm = jm.astype(np.int64)  # int(jm)
        f = jax.lax.select_n(2 * (ifp == ifm) + (ifp < ifm), ifm + 8, ifp, (ifp | 4))
        return T_hpc(jm - ifm, 1 + ifp - jp, f)

    def loc2hpc_caps(loc, tt):
        ntt = jnp.minimum(tt.astype(np.int64), 3)

        tp = tt - ntt  # [0;1)
        tmp = loc.s / jnp.sqrt((1.0 + jnp.abs(loc.z)) * (1.0 / 3.0))

        jp = tp * tmp  # increasing edge line index
        jm = (1.0 - tp) * tmp  # decreasing edge line index
        jp = jnp.minimum(jp, 1.0)  # for points too close to the boundary
        jm = jnp.minimum(jm, 1.0)
        cond = loc.z >= 0
        x = jax.lax.select(cond, 1.0 - jm, jp)
        y = jax.lax.select(cond, 1.0 - jp, jm)
        f = jax.lax.select(cond, ntt, ntt + 8)
        return T_hpc(x, y, f)

    tt = 4.0 * ((loc.phi * (1.0 / (2.0 * PI))) % 1)
    return jax.lax.cond(jnp.abs(loc.z) <= 2.0 / 3.0, loc2hpc_mid, loc2hpc_caps, loc, tt)


def hpc2loc(hpc: T_hpc) -> T_loc:
    def c0(jr, hpc):
        tmp = jr * jr * (1.0 / 3.0)
        phi = (PI * 0.25) * (JPLL[hpc.f] + (hpc.x - hpc.y) / jr)
        return T_loc(1.0 - tmp, jnp.sqrt(tmp * (2.0 - tmp)), phi)

    def c1(jr, hpc):
        jr = 4.0 - jr
        tmp = jr * jr * (1.0 / 3.0)
        phi = (PI * 0.25) * (JPLL[hpc.f] + (hpc.x - hpc.y) / jr)
        return T_loc(tmp - 1.0, jnp.sqrt(tmp * (2.0 - tmp)), phi)

    def c2(jr, hpc):
        z = (2.0 - jr) * (2.0 / 3.0)
        phi = (PI * 0.25) * (JPLL[hpc.f] + hpc.x - hpc.y)
        return T_loc(z, jnp.sqrt((1.0 + z) * (1.0 - z)), phi)

    def s2(jr, hpc):
        return jax.lax.cond(jr > 3.0, c1, c2, jr, hpc)

    jr = JRLL[hpc.f] - hpc.x - hpc.y
    return jax.lax.cond(jr < 1.0, c0, s2, jr, hpc)


def ang2loc(ang: T_ang) -> T_loc:
    cth, sth = jnp.cos(ang.theta), jnp.sin(ang.theta)
    cond = sth < 0
    phi = jax.lax.select(cond, ang.phi + PI, ang.phi)
    sth = jax.lax.select(cond, -sth, sth)
    return T_loc(z=cth, s=sth, phi=phi)


def loc2ang(loc: T_loc) -> T_ang:
    return T_ang(theta=jnp.arctan2(loc.s, loc.z), phi=loc.phi)


def vec2loc(vec: T_vec) -> T_loc:
    vlen = vec.norm()
    cth = vec.z / vlen
    sth = jnp.sqrt(vec.x**2 + vec.y**2) / vlen
    return T_loc(z=cth, s=sth, phi=jnp.arctan2(vec.y, vec.x))


def loc2vec(loc: T_loc) -> T_vec:
    return T_vec(x=loc.s * jnp.cos(loc.phi), y=loc.s * jnp.sin(loc.phi), z=loc.z)


def _ang2vec(ang: T_ang) -> T_vec:
    return loc2vec(ang2loc(ang))


def _vec2ang(vec: T_vec) -> T_ang:
    return T_ang(
        theta=jnp.atan2(jnp.sqrt(vec.x * vec.x + vec.y * vec.y), vec.z),
        phi=jnp.atan2(vec.y, vec.x),
    )


# Conversions between discrete coordinate systems


def isqrt(v: np.int64):
    res = jnp.sqrt(v + 0.5).astype(np.int64)

    def r2(res, v):
        res = jax.lax.select(res * res > v, res - 1, res)
        res = jax.lax.select(((res + 1) * (res + 1)) <= v, res + 1, res)
        return res

    return jax.lax.cond(v < (1 << 50), lambda x, v: x, r2, res, v)


def c_idiv(a, b):
    # mimic C like integer division int(a / b)
    res = a / b
    return int(res) if isinstance(res, float) else res.astype(np.int64)


def spread_bits(v: np.int64):
    res = v & 0xFFFFFFFF
    res = (res ^ (res << 16)) & 0x0000FFFF0000FFFF
    res = (res ^ (res << 8)) & 0x00FF00FF00FF00FF
    res = (res ^ (res << 4)) & 0x0F0F0F0F0F0F0F0F
    res = (res ^ (res << 2)) & 0x3333333333333333
    res = (res ^ (res << 1)) & 0x5555555555555555
    return res


def compress_bits(v: np.int64):
    res = v & 0x5555555555555555
    res = (res ^ (res >> 1)) & 0x3333333333333333
    res = (res ^ (res >> 2)) & 0x0F0F0F0F0F0F0F0F
    res = (res ^ (res >> 4)) & 0x00FF00FF00FF00FF
    res = (res ^ (res >> 8)) & 0x0000FFFF0000FFFF
    res = (res ^ (res >> 16)) & 0x00000000FFFFFFFF
    return res


def hpd2nest(nside: np.int64, hpd: T_hpd) -> np.int64:
    return (hpd.f * nside * nside) + spread_bits(hpd.x) + (spread_bits(hpd.y) << 1)


def nest2hpd(nside: np.int64, pix: np.int64) -> T_hpd:
    npface_ = nside * nside
    p2 = pix & (npface_ - 1)
    return T_hpd(x=compress_bits(p2), y=compress_bits(p2 >> 1), f=c_idiv(pix, npface_))


def hpd2ring(nside_: np.int64, hpd: T_hpd) -> np.int64:
    jr = (JRLL[hpd.f] * nside_) - hpd.x - hpd.y - 1

    def bound(x, b):
        return jax.lax.select_n(
            2 * (x > b) + (x < 1),
            x,
            x + b,
            x - b,
        )

    def c0(hpd, jr, nside):
        jp = c_idiv(JPLL[hpd.f] * jr + hpd.x - hpd.y + 1, 2)
        jp = bound(jp, 4 * nside)
        return 2 * jr * (jr - 1) + jp - 1

    def c1(hpd, jr, nside):
        nl4 = 4 * nside
        jri = nl4 - jr
        jp = c_idiv(JPLL[hpd.f] * jri + hpd.x - hpd.y + 1, 2)
        jp = bound(jp, nl4)
        return 12 * nside * nside - 2 * (jri + 1) * jri + jp - 1

    def c2(hpd, jr, nside):
        nl4 = 4 * nside
        jp = c_idiv(JPLL[hpd.f] * nside + hpd.x - hpd.y + 1 + ((jr - nside) & 1), 2)
        jp = bound(jp, nl4)
        return 2 * nside * (nside - 1) + (jr - nside) * nl4 + jp - 1

    def s2(hpd, jr, nside):
        return jax.lax.cond(jr > 3 * nside_, c1, c2, hpd, jr, nside)

    return jax.lax.cond(jr < nside_, c0, s2, hpd, jr, nside_)


def ring2hpd(nside: np.int64, pix: np.int64):
    ncap = 2 * nside * (nside - 1)
    npix = 12 * nside * nside

    def c0(pix, nside, ncap, npix):  # North Polar cap
        iring = (1 + isqrt(1 + 2 * pix)) >> 1  # counted from North pole
        iphi = (pix + 1) - 2 * iring * (iring - 1)
        face = c_idiv(iphi - 1, iring)
        irt = iring - (JRLL[face] * nside) + 1
        ipt = 2 * iphi - JPLL[face] * iring - 1
        ipt = jax.lax.select(ipt >= 2 * nside, ipt - 8 * nside, ipt)
        return T_hpd((ipt - irt) >> 1, (-(ipt + irt)) >> 1, face)

    def c1(pix, nside, ncap, npix):  # Equatorial region
        ip = pix - ncap
        iring = (c_idiv(ip, 4 * nside)) + nside  # counted from North pole
        iphi = (ip % (4 * nside)) + 1
        kshift = (iring + nside) & 1
        ire = iring - nside + 1
        irm = 2 * nside + 2 - ire
        ifm = c_idiv(iphi - c_idiv(ire, 2) + nside - 1, nside)
        ifp = c_idiv(iphi - c_idiv(irm, 2) + nside - 1, nside)
        face = jax.lax.select_n(2 * (ifp == ifm) + (ifp < ifm), ifm + 8, ifp, ifp | 4)
        irt = iring - (JRLL[face] * nside) + 1
        ipt = 2 * iphi - JPLL[face] * nside - kshift - 1
        ipt = jax.lax.select(ipt >= 2 * nside, ipt - 8 * nside, ipt)
        return T_hpd((ipt - irt) >> 1, (-(ipt + irt)) >> 1, face)

    def c2(pix, nside, ncap, npix):  # South Polar cap
        ip = npix - pix
        iring = (1 + isqrt(2 * ip - 1)) >> 1  # counted from South pole
        iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1))
        face = 8 + c_idiv(iphi - 1, iring)
        irt = 4 * nside - iring - (JRLL[face] * nside) + 1
        ipt = 2 * iphi - JPLL[face] * iring - 1
        ipt = jax.lax.select(ipt >= 2 * nside, ipt - 8 * nside, ipt)
        return T_hpd((ipt - irt) >> 1, (-(ipt + irt)) >> 1, face)

    def s2(pix, nside, ncap, npix):
        return jax.lax.cond(pix < (npix - ncap), c1, c2, pix, nside, ncap, npix)

    return jax.lax.cond(pix < ncap, c0, s2, pix, nside, ncap, npix)


def nest2ring(nside: np.int64, ipnest: np.int64) -> np.int64:
    if (nside & (nside - 1)) != 0:
        return -1
    return hpd2ring(nside, nest2hpd(nside, ipnest))


def ring2nest(nside: np.int64, ipring: np.int64) -> np.int64:
    if (nside & (nside - 1)) != 0:
        return -1
    return hpd2nest(nside, ring2hpd(nside, ipring))


# Mixed conversions


def loc2hpd(nside_: np.int64, loc: T_loc) -> T_hpd:
    tmp = loc2hpc(loc)
    return T_hpd(
        (tmp.x * nside_).astype(jnp.int64),
        (tmp.y * nside_).astype(jnp.int64),
        tmp.f,
    )


def hpd2loc(nside_: np.int64, hpd: T_hpd) -> T_loc:
    xns = 1.0 / nside_
    tmp = T_hpc(x=(hpd.x + 0.5) * xns, y=(hpd.y + 0.5) * xns, f=hpd.f)
    return hpc2loc(tmp)


def npix2nside(npix: np.int64) -> np.int64:
    res = isqrt(npix // 12)
    if res * res * 12 != npix:
        return -1
    return res


def nside2npix(nside: np.int64):
    return 12 * nside * nside


def ang2ring(nside: np.int64, ang: T_ang) -> np.int64:
    return hpd2ring(nside, loc2hpd(nside, ang2loc(ang)))


def ang2nest(nside: np.int64, ang: T_ang) -> np.int64:
    return hpd2nest(nside, loc2hpd(nside, ang2loc(ang)))


def vec2ring(nside: np.int64, vec: T_vec) -> np.int64:
    return hpd2ring(nside, loc2hpd(nside, vec2loc(vec)))


def vec2nest(nside: np.int64, vec: T_vec) -> np.int64:
    return hpd2nest(nside, loc2hpd(nside, vec2loc(vec)))


def ring2ang(nside: np.int64, ipix: np.int64) -> T_ang:
    return loc2ang(hpd2loc(nside, ring2hpd(nside, ipix)))


def nest2ang(nside: np.int64, ipix: np.int64) -> T_ang:
    return loc2ang(hpd2loc(nside, nest2hpd(nside, ipix)))


def ring2vec(nside: np.int64, ipix: np.int64) -> T_vec:
    return loc2vec(hpd2loc(nside, ring2hpd(nside, ipix)))


def nest2vec(nside: np.int64, ipix: np.int64) -> T_vec:
    return loc2vec(hpd2loc(nside, nest2hpd(nside, ipix)))


def nside2order(nside: np.int64) -> np.int64:
    # Round as jnp.log may use inconsistent arithmetic
    return jax.lax.select(
        (nside & (nside - 1)) == 0, jnp.round(jnp.log2(nside)).astype(jnp.int64), -1
    )


def face_neighbors_nest(nside, hpd):
    fpix = hpd.f << (2 * nside2order(nside))
    px0 = spread_bits(hpd.x)
    py0 = spread_bits(hpd.y) << 1
    pxp = spread_bits(hpd.x + 1)
    pyp = spread_bits(hpd.y + 1) << 1
    pxm = spread_bits(hpd.x - 1)
    pym = spread_bits(hpd.y - 1) << 1

    result = jnp.array(
        [
            fpix + pxm + py0,
            fpix + pxm + pyp,
            fpix + px0 + pyp,
            fpix + pxp + pyp,
            fpix + pxp + py0,
            fpix + pxp + pym,
            fpix + px0 + pym,
            fpix + pxm + pym,
        ],
        dtype=np.int64,
    )
    return result


def face_neighbors_ring(nside, hpd):
    result = list(
        hpd2ring(
            nside, T_hpd(x=hpd.x + NB_XOFFSET[m], y=hpd.y + NB_YOFFSET[m], f=hpd.f)
        )
        for m in range(8)
    )
    return jnp.array(result)


def edge_neighbors(nside, hpd, fun):
    def bound(xx, num, shift):
        cond = 2 * (xx < 0) + (xx >= nside)
        xx = jax.lax.select_n(cond, xx, xx - nside, xx + nside)
        num = jax.lax.select_n(cond, num, num + shift, num - shift)
        return xx, num

    result = []
    for i in range(8):
        x = hpd.x + NB_XOFFSET[i]
        y = hpd.y + NB_YOFFSET[i]
        nbnum = 4
        x, nbnum = bound(x, nbnum, 1)
        y, nbnum = bound(y, nbnum, 3)

        f = NB_FACEARRAY[nbnum, hpd.f]

        def nb():
            bits = NB_SWAPARRAY[nbnum, hpd.f >> 2]
            xx = jax.lax.select(bits & 1, nside - x - 1, x)
            yy = jax.lax.select(bits & 2, nside - y - 1, y)
            cc = bits & 4
            tm = xx
            xx = jax.lax.select(cc, yy, xx)
            yy = jax.lax.select(cc, tm, yy)
            return fun(nside, T_hpd(xx, yy, f))

        result.append(jax.lax.select(f >= 0, nb(), -1))
    return jnp.array(result)


def neighbors(nside, pix, nest=False):
    hpd = nest2hpd(nside, pix) if nest else ring2hpd(nside, pix)

    face = face_neighbors_nest if nest else face_neighbors_ring
    edge = partial(edge_neighbors, fun=hpd2nest if nest else hpd2ring)

    nsm1 = nside - 1
    cond = (hpd.x > 0) * (hpd.x < nsm1) * (hpd.y > 0) * (hpd.y < nsm1)
    return jax.lax.cond(cond, face, edge, nside, hpd)


# Mimic healpy function signatures


def pix2ang(nside, ipix, nest=False, lonlat=False):
    if lonlat:
        msg = "Only co-latitude and longitude in radians supported for now!"
        raise NotImplementedError(msg)
    ang = nest2ang(nside, ipix) if nest else ring2ang(nside, ipix)
    return ang.theta, ang.phi


def pix2vec(nside, ipix, nest=False):
    vec = nest2vec(nside, ipix) if nest else ring2vec(nside, ipix)
    return vec.x, vec.y, vec.z


def ang2pix(nside, theta, phi, nest=False, lonlat=False):
    if lonlat:
        msg = "Only co-latitude and longitude in radians supported for now!"
        raise NotImplementedError(msg)
    ang = T_ang(theta=theta, phi=phi)
    return ang2nest(nside, ang) if nest else ang2ring(nside, ang)


def vec2pix(nside, x, y, z, nest=False):
    vec = T_vec(x=x, y=y, z=z)
    return vec2nest(nside, vec) if nest else vec2ring(nside, vec)


def vec2ang(x, y, z, lonlat=False):
    if lonlat:
        msg = "Only co-latitude and longitude in radians supported for now!"
        raise NotImplementedError(msg)
    vec = T_vec(x=x, y=y, z=z)
    ang = _vec2ang(vec)
    return ang.theta, ang.phi


def ang2vec(theta, phi, lonlat=False):
    if lonlat:
        msg = "Only co-latitude and longitude in radians supported for now!"
        raise NotImplementedError(msg)
    ang = T_ang(theta, phi)
    vec = _ang2vec(ang)
    return vec.x, vec.y, vec.z


def get_all_neighbours(nside, theta_or_pix, phi=None, nest=False, lonlat=False):
    if lonlat:
        msg = "Only co-latitude and longitude in radians supported for now!"
        raise NotImplementedError(msg)
    if phi is not None:
        theta_or_pix = ang2pix(nside, theta_or_pix, phi, nest=nest, lonlat=lonlat)
    return neighbors(nside, theta_or_pix, nest=nest)


def get_all_neighbours_valid(nside, theta_or_pix, phi=None, nest=False, lonlat=False):
    nbrs = get_all_neighbours(nside, theta_or_pix, phi=phi, nest=nest, lonlat=lonlat)
    if not nest:
        nbrs = jax.vmap(ring2nest, (None, 0))(nside, nbrs)
    tmp = nbrs[BAD_NBR_SLCT] + BAD_NBR_SHFT
    nbrs = jax.lax.select(nbrs != -1, nbrs, tmp)
    if not nest:
        nbrs = jax.vmap(nest2ring, (None, 0))(nside, nbrs)
    return nbrs
