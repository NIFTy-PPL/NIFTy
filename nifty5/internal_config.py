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
# Copyright(C) 2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.


# Internal configuration switches, typically for experimental features.
# Leave unchanged unless you know what you are doing!

def parallelization_scheme():
    """Sets the MPI parallelization scheme according to the
    environment variable `NIFTY_MPI_SCHEME`.

    If not set, "Standard" parallelization is used.

    Possible Values:
    ---------------

    "Standard": Fields are distributed over all MPI instances
                along their first axis.
                This mode is useful if the fields involved are
                too large for the memory of a single machine,
                otherwise it is not recommended.

    "Samples":  :class:`MetricGaussianKL_MPI` becomes available.
                The :class:`MetricGaussianKL` usually has
                multiple samples for which it needs to perform
                the same calculations.
                :class:`MetricGaussianKL_MPI` distributes these
                calculations to the MPI instances by sample.
                This mode is useful when the "Standard" mode
                is not needed and the calculations w.r.t. each
                sample take long w.r.t. the MPI communication
                overhead introduced by the parallelization.

    "None":     Disables all parallelization.
    """
    import os
    scheme = os.getenv("NIFTY_MPI_SCHEME", default="Standard")
    if scheme not in ["Standard", "Samples", "None"]:
        raise ValueError("Unrecognized MPI parallelization scheme!")
    return scheme
