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
# Copyright(C) 2013-2021 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import operator
from collections import defaultdict

from ..sugar import domain_union
from ..utilities import indent
from .block_diagonal_operator import BlockDiagonalOperator
from .linear_operator import LinearOperator


class SumOperator(LinearOperator):
    """Class representing sums of operators.

    Notes
    -----
    This operator has to be called using the `make` method.
    """

    def __init__(self, ops, neg, dom, tgt, _callingfrommake=False):
        if not _callingfrommake:
            raise NotImplementedError
        self._ops = ops
        self._neg = neg
        self._domain = dom
        self._target = tgt
        self._capability = self.TIMES | self.ADJOINT_TIMES
        for op in ops:
            self._capability &= op.capability

        try:
            from ..re import unite

            def joined_jax_expr(x):
                res = None
                for op, n in zip(ops, neg):
                    tmp = op.jax_expr(x)
                    if res is None:
                        res = -tmp if n is True else tmp
                    else:
                        o = operator.sub if n is True else operator.add
                        res = unite(res, tmp, op=o)
                return res

            self._jax_expr = joined_jax_expr
        except ImportError:
            self._jax_expr = None

    @staticmethod
    def simplify(ops, neg):
        from .diagonal_operator import DiagonalOperator
        from .scaling_operator import ScalingOperator

        # unpack SumOperators
        opsnew = []
        negnew = []
        for op, ng in zip(ops, neg):
            if isinstance(op, SumOperator):
                opsnew += op._ops
                if ng:
                    negnew += [not n for n in op._neg]
                else:
                    negnew += list(op._neg)
# FIXME: this needs some more work to keep the domain and target unchanged!
#            elif isinstance(op, NullOperator):
#                pass
            else:
                opsnew.append(op)
                negnew.append(ng)
        ops = opsnew
        neg = negnew

        # sort operators according to domains
        sorted = defaultdict(list)
        for op, ng in zip(ops, neg):
            sorted[(op.domain, op.target)].append((op, ng))

        xxops = []
        xxneg = []
        for opset in sorted.values():
            # collect ScalingOperators
            sum = 0.
            opsnew = []
            negnew = []

            dtype = []
            for op, ng in opset:
                if isinstance(op, ScalingOperator):
                    sum += op._factor * (-1 if ng else 1)
                    dtype.append(op._dtype)
                else:
                    opsnew.append(op)
                    negnew.append(ng)
            lastdom = opset[0][0].domain
            del(opset)

            if len(dtype) > 0:
                # Propagate sampling dtypes only if they are all the same
                if all(dtype[0] == ss for ss in dtype):
                    dtype = dtype[0]
                else:
                    dtype = None

            if sum != 0.:
                # try to absorb the factor into a DiagonalOperator
                for i in range(len(opsnew)):
                    if isinstance(opsnew[i], DiagonalOperator):
                        if opsnew[i]._dtype != dtype:
                            continue
                        sum *= (-1 if negnew[i] else 1)
                        opsnew[i] = opsnew[i]._add(sum)
                        sum = 0.
                        break
            if sum != 0 or len(opsnew) == 0:
                # have to add the scaling operator at the end
                opsnew.append(ScalingOperator(lastdom, sum, dtype))
                negnew.append(False)
            del(dtype, sum, lastdom)

            ops = opsnew
            neg = negnew
            # Step 4: combine DiagonalOperators where possible
            processed = [False] * len(ops)
            opsnew = []
            negnew = []
            for i in range(len(ops)):
                if not processed[i]:
                    if isinstance(ops[i], DiagonalOperator):
                        op = ops[i]
                        opneg = neg[i]
                        for j in range(i+1, len(ops)):
                            if isinstance(ops[j], DiagonalOperator) and ops[i]._dtype == ops[j]._dtype:
                                op = op._combine_sum(ops[j], opneg, neg[j])
                                opneg = False
                                processed[j] = True
                        opsnew.append(op)
                        negnew.append(opneg)
                    else:
                        opsnew.append(ops[i])
                        negnew.append(neg[i])
            ops = opsnew
            neg = negnew

            # combine BlockDiagonalOperators where possible
            processed = [False] * len(ops)
            opsnew = []
            negnew = []
            for i in range(len(ops)):
                if not processed[i]:
                    if isinstance(ops[i], BlockDiagonalOperator):
                        op = ops[i]
                        opneg = neg[i]
                        for j in range(i+1, len(ops)):
                            if isinstance(ops[j], BlockDiagonalOperator):
                                op = op._combine_sum(ops[j], opneg, neg[j])
                                opneg = False
                                processed[j] = True
                        opsnew.append(op)
                        negnew.append(opneg)
                    else:
                        opsnew.append(ops[i])
                        negnew.append(neg[i])
            xxops += opsnew
            xxneg += negnew

        dom = domain_union([op.domain for op in xxops])
        tgt = domain_union([op.target for op in xxops])
        return xxops, xxneg, dom, tgt

    @staticmethod
    def make(ops, neg):
        """Build a SumOperator (or something simpler if possible)

        Parameters
        ----------
        ops: list of LinearOperator
            Individual operators of the sum.
        neg: list of bool
            Same length as ops.
            If True then the corresponding operator gets a minus in the sum.
        """
        ops = tuple(ops)
        neg = tuple(neg)
        if len(ops) == 0:
            raise ValueError("ops is empty")
        if len(ops) != len(neg):
            raise ValueError("length mismatch between ops and neg")
        ops, neg, dom, tgt = SumOperator.simplify(ops, neg)
        if len(ops) == 1:
            return -ops[0] if neg[0] else ops[0]
        return SumOperator(ops, neg, dom, tgt, _callingfrommake=True)

    @property
    def adjoint(self):
        return self.make([op.adjoint for op in self._ops], self._neg)

    def apply(self, x, mode):
        self._check_mode(mode)
        res = None
        for op, neg in zip(self._ops, self._neg):
            tmp = op.apply(x.extract(op._dom(mode)), mode)
            if res is None:
                res = -tmp if neg else tmp
            else:
                res = res.flexible_addsub(tmp, neg)
        return res

    def draw_sample(self, from_inverse=False):
        if from_inverse:
            raise NotImplementedError(
                "cannot draw from inverse of this operator")
        res = None
        for op in self._ops:
            from .simple_linear_operators import NullOperator
            if isinstance(op, NullOperator):
                continue
            tmp = op.draw_sample(from_inverse)
            res = tmp if res is None else res.unite(tmp)
        return res

    def __repr__(self):
        subs = "\n".join(sub.__repr__() for sub in self._ops)
        return "SumOperator:\n"+indent(subs)

    def _simplify_for_constant_input_nontrivial(self, c_inp):
        f = []
        o = []
        for op in self._ops:
            tf, to = op.simplify_for_constant_input(
                c_inp.extract_part(op.domain))
            f.append(tf)
            o.append(to)

        from ..multi_domain import MultiDomain
        if not isinstance(self._target, MultiDomain):
            fullop = None
            for to, n in zip(o, self._neg):
                op = to if not n else -to
                fullop = op if fullop is None else fullop + op
            return None, fullop

        from .simplify_for_const import ConstCollector
        cc = ConstCollector()
        fullop = None
        for tf, to, n in zip(f, o, self._neg):
            cc.add(tf, to.target)
            op = to if not n else -to
            fullop = op if fullop is None else fullop + op
        return cc.constfield, fullop
