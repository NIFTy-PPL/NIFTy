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

from .field import Field
from .operators.endomorphic_operator import EndomorphicOperator
from .operators.operator import Operator


class StatCalculator(object):
    """Helper class to compute mean and variance of a set of inputs.

    Notes
    -----
    - the memory usage of this object is constant, i.e. it does not increase
      with the number of samples added
    - the code computes the unbiased variance (which contains a `1./(n-1)`
      term for `n` samples).
    """
    def __init__(self):
        self._count = 0

    def add(self, value):
        """Adds a sample.

        Parameters
        ----------
        value: any type that supports multiplication by a scalar and
               element-wise addition/subtraction/multiplication.
        """
        self._count += 1
        if self._count == 1:
            self._mean = 1.*value
            self._M2 = 0.*value
        else:
            delta = value - self._mean
            self._mean = self.mean + delta*(1./self._count)
            delta2 = value - self._mean
            self._M2 = self._M2 + delta*delta2

    @property
    def mean(self):
        """
        value type : the mean of all samples added so far.
        """
        if self._count == 0:
            raise RuntimeError
        return 1.*self._mean

    @property
    def var(self):
        """
        value type : the unbiased variance of all samples added so far.
        """
        if self._count < 2:
            raise RuntimeError
        return self._M2 * (1./(self._count-1))


def probe_with_posterior_samples(op, post_op, nprobes):
    '''FIXME

    Parameters
    ----------
    op : EndomorphicOperator
        FIXME
    post_op : Operator
        FIXME
    nprobes : int
        Number of samples which shall be drawn.

    Returns
    -------
    List of Field
        List of two fields: the mean and the variance.
    '''
    if not isinstance(op, EndomorphicOperator):
        raise TypeError
    if post_op is not None:
        if not isinstance(post_op, Operator):
            raise TypeError
        if post_op.domain is not op.target:
            raise ValueError
    sc = StatCalculator()
    for i in range(nprobes):
        if post_op is None:
            sc.add(op.draw_sample(from_inverse=True))
        else:
            sc.add(post_op(op.draw_sample(from_inverse=True)))

    if nprobes == 1:
        return sc.mean, None
    return sc.mean, sc.var


def probe_diagonal(op, nprobes, random_type="pm1"):
    '''Probes the diagonal of an endomorphic operator.

    The operator is called on a user-specified number of randomly generated
    input vectors :math:`v_i`, producing :math:`r_i`. The estimated diagonal
    is the mean of :math:`r_i^\\dagger v_i`.

    Parameters
    ----------
    op: EndomorphicOperator
        The operator to be probed.
    nprobes: int
        The number of probes to be used.
    random_type: str
        The kind of random number distribution to be used for the probing.
        The default value `pm1` causes the probing vector to be randomly
        filled with values of +1 and -1.

    Returns
    -------
    Field
        The estimated diagonal.
    '''
    sc = StatCalculator()
    for i in range(nprobes):
        input = Field.from_random(random_type, op.domain)
        output = op(input)
        sc.add(output.conjugate()*input)
    return sc.mean
