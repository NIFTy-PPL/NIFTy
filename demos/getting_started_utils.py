import numpy as np

import nifty8 as ift


class SingleDomain(ift.LinearOperator):
    def __init__(self, domain, target):
        self._domain = ift.makeDomain(domain)
        self._target = ift.makeDomain(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        return ift.makeField(self._tgt(mode), x.val)


class IRGSpace(ift.StructuredDomain):
    """Represents non-equidistantly binned and non-periodic one-dimensional spaces.

    Parameters
    ----------
    coordinates : np.ndarray
        Must be sorted and strictly ascending.
    """

    _needed_for_hash = ["_coordinates"]

    def __init__(self, coordinates):
        bb = np.array(coordinates)
        if bb.ndim != 1:
            raise TypeError
        if np.any(np.diff(bb) <= 0.0):
            raise ValueError("Coordinates must be sorted and strictly ascending")
        self._coordinates = tuple(bb)

    def __repr__(self):
        return f"IRGSpace(shape={self.shape}, coordinates={self._coordinates})"

    @property
    def harmonic(self):
        """bool : Always False for this class."""
        return False

    @property
    def shape(self):
        return (len(self._coordinates),)

    @property
    def size(self):
        return self.shape[0]

    @property
    def scalar_dvol(self):
        return None

    @property
    def dvol(self):
        """Assume that the coordinates are the center of symmetric pixels."""
        return np.diff(self.binbounds())

    def binbounds(self):
        if len(self._coordinates) == 1:
            return np.array([-np.inf, np.inf])
        c = np.array(self._coordinates)
        bounds = np.empty(self.size + 1)
        bounds[1:-1] = c[:-1] + 0.5*np.diff(c)
        bounds[0] = c[0] - 0.5*(c[1] - c[0])
        bounds[-1] = c[-1] + 0.5*(c[-1] - c[-2])
        return bounds

    @property
    def distances(self):
        return np.diff(self._coordinates)

    @property
    def coordinates(self):
        return self._coordinates


class WienerIntegrations(ift.LinearOperator):
    """Operator that performs the integrations necessary for an integrated
    Wiener process.

    Parameters
    ----------
    time_domain : IRGSpace
        Domain that contains the temporal information of the process.

    remaining_domain : DomainTuple or Domain
        All integrations are handled independently for this domain.
    """
    def __init__(self, time_domain, remaining_domain):
        self._target = ift.makeDomain((time_domain, remaining_domain))
        dom = ift.UnstructuredDomain((2, time_domain.size - 1)), remaining_domain
        self._domain = ift.makeDomain(dom)
        self._volumes = time_domain.distances
        for _ in range(len(remaining_domain.shape)):
            self._volumes = self._volumes[..., np.newaxis]
        self._capability = self.TIMES | self.ADJOINT_TIMES


    def apply(self, x, mode):
        self._check_input(x, mode)
        first, second = (0,), (1,)
        from_second = (slice(1, None),)
        no_border = (slice(0, -1),)
        reverse = (slice(None, None, -1),)
        if mode == self.TIMES:
            x = x.val
            res = np.zeros(self._target.shape)
            res[from_second] = np.cumsum(x[second], axis=0)
            res[from_second] = (res[from_second] + res[no_border]) / 2 * self._volumes + x[first]
            res[from_second] = np.cumsum(res[from_second], axis=0)
        else:
            x = x.val_rw()
            res = np.zeros(self._domain.shape)
            x[from_second] = np.cumsum(x[from_second][reverse], axis=0)[reverse]
            res[first] += x[from_second]
            x[from_second] *= self._volumes / 2.0
            x[no_border] += x[from_second]
            res[second] += np.cumsum(x[from_second][reverse], axis=0)[reverse]
        return ift.makeField(self._tgt(mode), res)


def IntWProcessInitialConditions(a0, b0, wpop, irg_space=None):
    if ift.is_operator(wpop):
        tgt = wpop.target
    else:
        tgt = irg_space, a0.target[0]

    sdom = tgt[0]

    bc = _FancyBroadcast(tgt)
    factors = ift.full(sdom, 0)
    factors = np.empty(sdom.shape)
    factors[0] = 0
    factors[1:] = np.cumsum(sdom.distances)
    factors = ift.makeField(sdom, factors)
    res = bc @ a0 + ift.DiagonalOperator(factors, tgt, 0) @ bc @ b0

    if wpop is None:
        return res
    else:
        return wpop + res


class _FancyBroadcast(ift.LinearOperator):
    def __init__(self, target):
        self._target = ift.DomainTuple.make(target)
        self._domain = ift.DomainTuple.make(target[1])
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            res = np.broadcast_to(x.val[None], self._target.shape)
        else:
            res = np.sum(x.val, axis=0)
        return ift.makeField(self._tgt(mode), res)


class DomainChangerAndReshaper(ift.LinearOperator):
    def __init__(self, domain, target):
        self._domain = ift.DomainTuple.make(domain)
        self._target = ift.DomainTuple.make(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        tgt = self._tgt(mode)
        return ift.makeField(tgt, x.reshape(tgt.shape))


def random_los(n_los):
    starts = list(ift.random.current_rng().random((n_los, 2)).T)
    ends = list(ift.random.current_rng().random((n_los, 2)).T)
    return starts, ends


def radial_los(n_los):
    starts = list(ift.random.current_rng().random((n_los, 2)).T)
    ends = list(0.5 + 0*ift.random.current_rng().random((n_los, 2)).T)
    return starts, ends