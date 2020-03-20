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
# Copyright(C) 2013-2020 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

# Stack of SeedSequence objects. Will always start out with a well-defined
# default. Users can change the "random seed" used by a calculation by pushing
# a different SeedSequence before invoking any other nifty6.random calls
_sseq = [np.random.SeedSequence(42)]
# Stack of random number generators associated with _sseq.
_rng = [np.random.default_rng(_sseq[-1])]


def spawn_sseq(n, parent=None):
    """Returns a list of `n` SeedSequence objects which are children of `parent`

    Parameters
    ----------
    n : int
        number of requested SeedSequence objects
    parent : SeedSequence
        the object from which the returned objects will be derived
        If `None`, the top of the current SeedSequence stack will be used

    Returns
    -------
    list(SeedSequence)
        the requested SeedSequence objects
    """
    if parent is None:
        global _sseq
        parent = _sseq[-1]
    return parent.spawn(n)


def current_rng():
    """Returns the RNG object currently in use by NIFTy

    Returns
    -------
    Generator
        the current Generator object (top of the generatir stack)
    """
    return _rng[-1]


def push_sseq(sseq):
    """Pushes a new SeedSequence object onto the SeedSequence stack.
    This also pushes a new Generator object built from the new SeedSequence
    to the generator stack.

    Parameters
    ----------
    sseq: SeedSequence
        the SeedSequence object to be used from this point
    """
    _sseq.append(sseq)
    _rng.append(np.random.default_rng(_sseq[-1]))


def push_sseq_from_seed(seed):
    """Pushes a new SeedSequence object derived from an integer seed onto the
    SeedSequence stack.
    This also pushes a new Generator object built from the new SeedSequence
    to the generator stack.

    Parameters
    ----------
    seed: int
        the seed from which the new SeedSequence will be built
    """
    _sseq.append(np.random.SeedSequence(seed))
    _rng.append(np.random.default_rng(_sseq[-1]))


def pop_sseq():
    """Pops the top of the SeedSequence and generator stacks."""
    _sseq.pop()
    _rng.pop()


class Random(object):
    @staticmethod
    def pm1(dtype, shape):
        if np.issubdtype(dtype, np.complexfloating):
            x = np.array([1+0j, 0+1j, -1+0j, 0-1j], dtype=dtype)
            x = x[_rng[-1].integers(0, 4, size=shape)]
        else:
            x = 2*_rng[-1].integers(0, 2, size=shape)-1
        return x.astype(dtype, copy=False)

    @staticmethod
    def normal(dtype, shape, mean=0., std=1.):
        if not (np.issubdtype(dtype, np.floating) or
                np.issubdtype(dtype, np.complexfloating)):
            raise TypeError("dtype must be float or complex")
        if not np.isscalar(mean) or not np.isscalar(std):
            raise TypeError("mean and std must be scalars")
        if np.issubdtype(type(std), np.complexfloating):
            raise TypeError("std must not be complex")
        if ((not np.issubdtype(dtype, np.complexfloating)) and
                np.issubdtype(type(mean), np.complexfloating)):
            raise TypeError("mean must not be complex for a real result field")
        if np.issubdtype(dtype, np.complexfloating):
            x = np.empty(shape, dtype=dtype)
            x.real = _rng[-1].normal(mean.real, std*np.sqrt(0.5), shape)
            x.imag = _rng[-1].normal(mean.imag, std*np.sqrt(0.5), shape)
        else:
            x = _rng[-1].normal(mean, std, shape).astype(dtype, copy=False)
        return x

    @staticmethod
    def uniform(dtype, shape, low=0., high=1.):
        if not np.isscalar(low) or not np.isscalar(high):
            raise TypeError("low and high must be scalars")
        if (np.issubdtype(type(low), np.complexfloating) or
                np.issubdtype(type(high), np.complexfloating)):
            raise TypeError("low and high must not be complex")
        if np.issubdtype(dtype, np.complexfloating):
            x = np.empty(shape, dtype=dtype)
            x.real = _rng[-1].uniform(low, high, shape)
            x.imag = _rng[-1].uniform(low, high, shape)
        elif np.issubdtype(dtype, np.integer):
            if not (np.issubdtype(type(low), np.integer) and
                    np.issubdtype(type(high), np.integer)):
                raise TypeError("low and high must be integer")
            x = _rng[-1].integers(low, high+1, shape)
        else:
            x = _rng[-1].uniform(low, high, shape)
        return x.astype(dtype, copy=False)
