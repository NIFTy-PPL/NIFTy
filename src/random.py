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

"""
Some remarks on NIFTy's treatment of random numbers

NIFTy makes use of the `Generator` and `SeedSequence` classes introduced to
`numpy.random` in numpy 1.17.

On first load of the `nifty8.random` module, it creates a stack of
`SeedSequence` objects which contains a single `SeedSequence` with a fixed seed,
and also a stack of `Generator` objects, which contains a single generator
derived from the above seed sequence. Without user intervention, this generator
will be used for all random number generation tasks within NIFTy. This means

- that random numbers drawn by NIFTy will be reproducible across multiple runs
  (assuming there are no complications like MPI-enabled runs with a varying
  number of tasks), and

- that trying to change random seeds via `numpy.random.seed` will have no
  effect on the random numbers drawn by NIFTy.

Users who want to change the random seed for a given run can achieve this
by calling :func:`push_sseq_from_seed()` with a seed of their choice. This will
push a new seed sequence generated from that seed onto the seed sequence stack,
and a generator derived from this seed sequence onto the generator stack.
Since all NIFTy RNG-related calls will use the generator on the top of the stack,
all calls from this point on will use the new generator.
If the user already has a `SeedSequence` object at hand, they can pass this to
NIFTy via :func:`push_sseq`. A new generator derived from this sequence will then
also be pushed onto the generator stack.
These operations can be reverted (and should be, as soon as the new generator is
no longer needed) by a call to :func:`pop_sseq()`.
When users need direct access to the RNG currently in use, they can access it
via the :func:`current_rng` function.


Example for using multiple seed sequences:

Assume that N samples are needed to compute a KL, which are distributed over
a variable number of MPI tasks. In this situation, whenever random numbers
need to be drawn for these samples:

- each MPI task should spawn as many seed sequences as there are samples
  *in total*, using ``sseq = spawn_sseq(N)``

- each task loops over the local samples

  - first pushing the seed sequence for the **global** index of the
    current sample via ``push_sseq(sseq[iglob])```

  - drawing the required random numbers

  - then popping the seed sequence again via ``pop_sseq()``

That way, random numbers should be reproducible and independent of the number
of MPI tasks.

WARNING: do not push/pop the same `SeedSequence` object more than once - this
will lead to repeated random sequences! Whenever you have to push `SeedSequence`
objects, generate new ones via :func:`spawn_sseq()`.
"""

import numpy as np


# Stack of SeedSequence objects. Will always start out with a well-defined
# default. Users can change the "random seed" used by a calculation by pushing
# a different SeedSequence before invoking any other nifty8.random calls
_sseq = [np.random.SeedSequence(42)]
# Stack of random number generators associated with _sseq.
_rng = [np.random.default_rng(_sseq[-1])]


def getState():
    """Returns the full internal state of the module. Intended for pickling.

    Returns
    -------
    state : unspecified
    """
    import pickle
    return pickle.dumps((_sseq, _rng))


def setState(state):
    """Restores the full internal state of the module. Intended for unpickling.


    Parameters
    ----------
    state : unspecified
        Result of an earlier call to `getState`.
    """
    import pickle
    global _sseq, _rng
    _sseq, _rng = pickle.loads(state)


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

    Notes
    -----
    This function should only be used

    - if you only want to change the random seed once at the very beginning
      of a run, or

    - if the restoring of the previous state has to happen in a different
      Python function. In this case, please make sure that there is a matching
      call to :func:`pop_sseq` for every call to this function!

    In all other situations, it is highly recommended to use the
    :class:`Context` class for managing the RNG state.
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

    Notes
    -----
    This function should only be used

    - if you only want to change the random seed once at the very beginning
      of a run, or

    - if the restoring of the previous state has to happen in a different
      Python function. In this case, please make sure that there is a matching
      call to :func:`pop_sseq` for every call to this function!

    In all other situations, it is highly recommended to use the
    :class:`Context` class for managing the RNG state.
    """
    _sseq.append(np.random.SeedSequence(seed))
    _rng.append(np.random.default_rng(_sseq[-1]))


def pop_sseq():
    """Pops the top of the SeedSequence and generator stacks."""
    _sseq.pop()
    _rng.pop()


class Random:
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
            x.real = _rng[-1].normal(mean.real, std, shape)
            x.imag = _rng[-1].normal(mean.imag, std, shape)
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


class Context:
    """Convenience class for easy management of the RNG state.
    Usage: ::

        with ift.random.Context(seed|sseq):
            code using the new RNG state

    At the end of the scope, the original RNG state will be restored
    automatically.

    Parameters
    ----------
    inp : int or numpy.random.SeedSequence
        The starting information for the new RNG state.
        If it is an integer, a new `SeedSequence` will be generated from it.
    """

    def __init__(self, inp):
        if not isinstance(inp, np.random.SeedSequence):
            inp = np.random.SeedSequence(inp)
        self._sseq = inp

    def __enter__(self):
        self._depth = len(_sseq)
        push_sseq(self._sseq)

    def __exit__(self, exc_type, exc_value, tb):
        pop_sseq()
        if self._depth != len(_sseq):
            raise RuntimeError("inconsistent RNG usage detected")
        return exc_type is None
