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
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras, Philipp Frank

import pickle

from .. import utilities
from ..multi_domain import MultiDomain
from ..multi_field import MultiField


class SampleList:
    """Base class for storing lists of fields representing samples.

    This class suits as a base class for storing lists of in most cases
    posterior samples. It is intended to be used to hold the minimization state
    of an inference run and comes with a variety of convenience functions like
    computing the mean or standard deviation of the output of a given operator
    over the sample list.

    Parameters
    ----------
    comm : MPI communicator or None
        If not `None`, :class:`SampleList` can gather samples across multiple
        MPI tasks. If `None`, :class:`SampleList` is not a distributed object.
    domain : Domainoid (can be DomainTuple, MultiDomain, dict, Domain or list of Domains)
        The domain on which the samples are defined.

    Note
    ----
    A class inheriting from :class:`SampleList` needs to call the constructor of
    `SampleList` and needs to implement :attr:`__len__()` and `__getitem__()`.
    """
    def __init__(self, comm, domain):
        from ..sugar import makeDomain
        self._comm = comm
        self._domain = makeDomain(domain)
        utilities.check_MPI_equality(self._domain, comm)

    def __len__(self):
        """int: Number of local samples."""
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError

    @property
    def comm(self):
        """MPI communicator or None: The communicator used for the SampleList."""
        return self._comm

    @property
    def domain(self):
        """DomainTuple or MultiDomain: the domain on which the samples are defined."""
        return self._domain

    @staticmethod
    def indices_from_comm(n_samples, comm):
        """Return range of global sample indices for local task.

        This method calls `utilities.shareRange`

        Parameters
        ----------
        n_samples : int
            Number of work items to be distributed.
        comm : MPI communicator or None
            The communicator used for the distribution.

        Returns
        -------
        range
            Range of relevant indices for the local task.
        """
        ntask, rank, _ = utilities.get_MPI_params_from_comm(comm)
        return range(*utilities.shareRange(n_samples, ntask, rank))

    def global_iterator(self, op=None):
        """Return iterator over all potentially distributed samples.

        Parameters
        ----------
        op : callable or None
            Callable that is applied to each item in the :class:`SampleList`
            before it is returned. Can be an
            :class:`~nifty8.operators.operator.Operator` or any other callable
            that takes a :class:`~nifty8.field.Field` as an input. Default:
            None.

        Note
        ----
        Calling this function involves MPI communication if `comm != None`.
        """
        op = _none_to_id(op)
        if self.comm is not None:
            for itask in range(self.comm.Get_size()):
                for i in range(_bcast(len(self), self._comm, itask)):
                    ss = self[i] if itask == self._comm.Get_rank() else None
                    yield op(_bcast(ss, self._comm, itask))
        else:
            for ss in self:
                yield op(ss)

    def global_average(self, op=None):
        """Compute average over all potentially distributed samples.

        Parameters
        ----------
        op : callable or None
            Callable that is applied to each item in the :class:`SampleList`
            before it is averaged. If `op` returns tuple, then individual
            averages are computed and returned individually as tuple.

        Note
        ----
        Calling this function involves MPI communication if `comm != None`.

        Note
        ----
        Calling this function involves allocating arrays for all samples and
        averaging them afterwards. If the number of local samples is big and
        `op` is not None, this leads to much temporary memory usage. If the
        output of `op` is just a :class:`~nifty8.field.Field` or
        :class:`~nifty8.multi_field.MultiField`, :attr:`global_sample_stat()`
        can be used in order to compute the average memory efficiently.
        """
        op = _none_to_id(op)
        res = [op(ss) for ss in self]
        n = self.global_n_samples()
        if not isinstance(res[0], tuple):
            return utilities.allreduce_sum(res, self.comm) / n
        n_output_elements = len(res[0])
        res = [[elem[ii] for elem in res] for ii in range(n_output_elements)]
        return tuple(utilities.allreduce_sum(rr, self.comm) / n for rr in res)

    def global_n_samples(self):
        """Return number of samples across all MPI tasks."""
        return utilities.allreduce_sum([len(self)], self.comm)

    def global_sample_stat(self, op=None):
        """Compute mean and variance of samples after applying `op`.

        Parameters
        ----------
        op : callable or None
            Callable that is applied to each item in the :class:`SampleList`
            before it is used to compute mean and variance.

        Returns
        -------
        tuple
            A tuple with two items: the mean and the variance.
        """
        from ..probing import StatCalculator
        sc = StatCalculator()
        for ss in self.global_iterator(op):
            sc.add(ss)
        return sc.mean, sc.var

    def save(self, file_name_base):
        """Serialize SampleList and write it to disk.

        Parameters
        ----------
        file_name_base : str
            File name of the output file without extension. The actual file name
            will have the extension ".pickle" and before that an identifier that
            distunguishes between MPI tasks.

        Note
        ----
        If the instance of :class:`SampleList` is distributed, each MPI task
        writes its own file.
        """
        fname = str(file_name_base) + _mpi_file_extension(self.comm) + ".pickle"
        with open(fname, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(file_name_base, comm=None):
        """Deserialize SampleList from files on disk.

        Parameters
        ----------
        file_name_base : str
            File name of the input file without extension. The actual file name
            will have the extension ".pickle" and before that an identifier that
            distunguishes between MPI tasks.
        comm : MPI communicator or None
            If not `None`, each MPI task reads its own input file.

        Note
        ----
        `file_name_base` needs to be the same string that has been used for
        saving the :class:`SampleList`.

        Note
        ----
        The number of MPI tasks used for saving and loading the `SampleList`
        need to be the same.
        """
        fname = str(file_name_base) + _mpi_file_extension(comm) + ".pickle"
        with open(fname, "rb") as f:
            obj = pickle.load(f)
        utilities.myassert(isinstance(obj, SampleList))
        return obj


class ResidualSampleList(SampleList):
    def __init__(self, mean, residuals, neg, comm):
        """SampleList that stores samples in terms of a mean and a residual deviation thereof.


        Parameters
        ----------
        mean : Field or MultiField
            Mean of the sample list.
        residuals : list of Field or list of MultiField
            List of residuals from the mean. If it is a list of `MultiField`,
            the domain of the residuals can be a subdomain of the domain of
            mean. This results in adding just a zero in respective `MultiField`
            entries.
        neg: list of bool
            This list has to have the same length as `residuals`. If an entry is
            `True`, the respective residual is subtracted and not added.
        comm : MPI communicator or None
            If not `None`, samples can be gathered across multiple MPI tasks. If
            `None`, :class:`ResidualSampleList` is not a distributed object.
        """
        super(ResidualSampleList, self).__init__(comm, mean.domain)
        self._m = mean
        self._r = tuple(residuals)
        self._n = tuple(neg)

        if len(self._r) != len(self._n):
            raise ValueError("Residuals and neg need to have the same length.")

        r_dom = self._r[0].domain
        if not all(rr.domain is r_dom for rr in self._r):
            raise ValueError("All residuals must have the same domain.")
        if isinstance(r_dom, MultiDomain):
            try:
                self._m.extract(r_dom)
            except:
                raise ValueError("`residual.domain` must be a subdomain of `mean.domain`.")

        if not all(isinstance(nn, bool) for nn in neg):
            raise TypeError("All entries in neg need to be bool.")

    def __getitem__(self, i):
        return self._m.flexible_addsub(self._r[i], self._n[i])

    def __len__(self):
        return len(self._r)

    def at_strict(self, mean):
        """Return a new instance of `ResidualSampleList` with new mean and the
        same residuals as `self`.

        The old and new mean need to be defined on the same domain.

        Returns
        -------
        ResidualSampleList
            Sample list with updated mean.
        """
        if mean.domain is not self.domain:
            raise ValueError("New and old mean have different domains:\n"
                             f"old: {self.domain}\n"
                             f"new: {mean.domain}\n")
        return self.at(mean)

    def at(self, mean):
        """Return a new instance of `ResidualSampleList` with new mean and the
        same residuals as `self`.

        Note
        ----
        If `self.domain` is a :class:`~nifty8.multi_domain.MultiDomain`, the old
        and new mean are combined with
        :attr:`~nifty8.multi_field.MultiField.union` beforehand. This means that
        only the multi field entries present in `mean` are updated.

        Returns
        -------
        ResidualSampleList
            Sample list with updated mean.
        """
        if isinstance(self._m, MultiField) and self.domain is not mean.domain:
            mean = MultiField.union([self._m, mean])
        return ResidualSampleList(mean, self._r, self._n, self.comm)


class MinimalSampleList(SampleList):
    def __init__(self, samples, comm=None):
        super(MinimalSampleList, self).__init__(comm, samples[0].domain)
        self._s = samples

    def __getitem__(self, x):
        return self._s[x]

    def __len__(self):
        return len(self._s)


def _none_to_id(obj):
    """If the input is None, replace it with identity map. Otherwise return
    input.
    """
    if obj is None:
        return lambda x: x
    return obj


def _bcast(obj, comm, root):
    """Broadcast python object from given root

    Parameters
    ----------
    obj : object
        The object to be broadcasted.
    comm : MPI communicator
        MPI communicator used for the broadcasting.
    root : int
        MPI task number from which the object shall be sent.
    """
    data = obj if comm.Get_rank() == root else None
    return comm.bcast(data, root=root)


def _mpi_file_extension(comm):
    """Return string that can be used to uniquely determine the number of MPI
    tasks for distributed saving of files.

    Parameters
    ----------
    comm : MPI communicator or None
        If None, an empty string is returned.
    """
    if comm is None:
        return ""
    ntask, rank, _ = utilities.get_MPI_params_from_comm(comm)
    return f"{rank}/{ntask}"
