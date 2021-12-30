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

from .. import utilities
from .diagonal_operator import DiagonalOperator
from .linear_operator import LinearOperator
from .scaling_operator import ScalingOperator
from .simple_linear_operators import NullOperator


class ChainOperator(LinearOperator):
    """Class representing chains of operators.

    Notes
    -----
    This operator has to be called using the :attr:`make` method.
    """

    def __init__(self, ops, _callingfrommake=False):
        if not _callingfrommake:
            raise NotImplementedError
        self._ops = ops
        self._capability = self._all_ops
        for op in ops:
            self._capability &= op.capability
        self._domain = self._ops[-1].domain
        self._target = self._ops[0].target

        if all(callable(op.jax_expr) for op in ops):

            def joined_jax_op(x):
                for op in reversed(ops):
                    x = op.jax_expr(x)
                return x

            self._jax_expr = joined_jax_op
        else:
            self._jax_expr = None

    @staticmethod
    def simplify(ops):
        # verify domains
        for i in range(len(ops) - 1):
            utilities.check_object_identity(ops[i + 1].target, ops[i].domain)
        # unpack ChainOperators
        opsnew = []
        for op in ops:
            opsnew += op._ops if isinstance(op, ChainOperator) else [op]
        ops = opsnew
        # check for NullOperators
        if any(isinstance(op, NullOperator) for op in ops):
            ops = (NullOperator(ops[-1].domain, ops[0].target),)
        # collect ScalingOperators
        fct = 1.
        opsnew = []
        lastdom = ops[-1].domain
        dtype = None
        for op in ops:
            if (isinstance(op, ScalingOperator) and op._factor.imag == 0):
                fct *= op._factor.real
            else:
                opsnew.append(op)
        if fct != 1.:
            # try to absorb the factor into a DiagonalOperator
            for i in range(len(opsnew)):
                if isinstance(opsnew[i], DiagonalOperator):
                    opsnew[i] = opsnew[i]._scale(fct)
                    fct = 1.
                    break
        if fct != 1 or len(opsnew) == 0:
            # have to add the scaling operator at the end
            op = ScalingOperator(lastdom, fct)
            if dtype is not None:
                op.dtype = dtype
            opsnew.append(op)
        ops = opsnew
        # combine DiagonalOperators where possible
        opsnew = []
        for op in ops:
            if (len(opsnew) > 0 and isinstance(opsnew[-1], DiagonalOperator)
                    and isinstance(op, DiagonalOperator)):
                opsnew[-1] = opsnew[-1]._combine_prod(op)
            else:
                opsnew.append(op)
        ops = opsnew
        # combine BlockDiagonalOperators where possible
        from .block_diagonal_operator import BlockDiagonalOperator
        opsnew = []
        for op in ops:
            if (len(opsnew) > 0
                    and isinstance(opsnew[-1], BlockDiagonalOperator)
                    and isinstance(op, BlockDiagonalOperator)):
                opsnew[-1] = opsnew[-1]._combine_chain(op)
            else:
                opsnew.append(op)
        ops = opsnew
        return ops

    @staticmethod
    def make(ops):
        """Build a ChainOperator (or something simpler if possible),
        a sequence of concatenated LinearOperators.

        Parameters
        ----------
        ops: list of LinearOperator
            Individual operators of the chain.
        """
        ops = tuple(ops)
        if len(ops) == 0:
            raise ValueError("ops is empty")
        ops = ChainOperator.simplify(ops)
        if len(ops) == 1:
            return ops[0]
        return ChainOperator(ops, _callingfrommake=True)

    def _flip_modes(self, trafo):
        ADJ = self.ADJOINT_BIT
        INV = self.INVERSE_BIT

        if trafo == 0:
            return self
        if trafo == ADJ or trafo == INV:
            return self.make(
                [op._flip_modes(trafo) for op in reversed(self._ops)])
        if trafo == ADJ | INV:
            return self.make([op._flip_modes(trafo) for op in self._ops])
        raise ValueError("invalid operator transformation")

    def apply(self, x, mode):
        self._check_mode(mode)
        t_ops = self._ops if mode & self._backwards else reversed(self._ops)
        for op in t_ops:
            x = op.apply(x, mode)
        return x

    def __repr__(self):
        subs = "\n".join(sub.__repr__() for sub in self._ops)
        return "ChainOperator:\n" + utilities.indent(subs)

    def _simplify_for_constant_input_nontrivial(self, c_inp):
        from ..multi_domain import MultiDomain
        if not isinstance(self._domain, MultiDomain):
            return None, self
        newop = None
        for op in reversed(self._ops):
            c_inp, t_op = op.simplify_for_constant_input(c_inp)
            newop = t_op if newop is None else op(newop)
        return c_inp, newop
