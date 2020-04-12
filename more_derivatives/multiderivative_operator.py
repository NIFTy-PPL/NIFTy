import nifty6 as ift
from differential_tensor import DiagonalTensor, SumTensor, GenLeibnizTensor
from multi_linearization import MultiLinearization



class MultiOperator:
    @property
    def domain(self):
        return self._domain
    @property
    def target(self):
        return self._target

    def __call__(self, x):
        return self.apply(x)

    def apply(self, x):
        raise NotImplementedError

class MultiLocalNonlin(MultiOperator):
    def __init__(self, domain, f, jacs):
        self._domain = ift.makeDomain(domain)
        self._target = ift.makeDomain(domain)
        self._f, self._jacs = f, jacs

    def apply(self, x):
        v = x.val
        res = self._f(v)
        jacs = [ift.makeOp(self._jacs[0](v)),]
        jacs += [DiagonalTensor(jac_i(v), i+2)
                 for i, jac_i in enumerate(self._jacs[1:])]
        return x.new(res, jacs)

class MultiLocalExp(MultiOperator):
    def __init__(self, domain):
        self._domain = self._target = ift.makeDomain(domain)

    def apply(self, x):
        v = x.val
        res = v.exp()
        jacs = [ift.makeOp(res), ]
        jacs += [DiagonalTensor(res, i) for i in range(2, x.maxderiv+1)]
        return x.new(res, jacs)

class MultiLinearOperator(MultiOperator):
    def __init__(self, op):
        self._domain = op.domain
        self._target = op.target
        self._op = op

    def apply(self, x):
        v = x.val
        res = self._op(v)
        jacs = [self._op, ] + [None, ]*(x.maxderiv-1)
        return x.new(res, jacs)

class MultiSumOperator(MultiOperator):
    def __init__(self, ops):
        self._target = ops[0].target
        for op in ops:
            assert self._target == op.target
        self._domain = ift.domain_union([op.domain for op in ops])
        self._ops = ops

    def apply(self, x):
        v = x.val

        j0 = None
        pjacs = []
        res = 0.
        for op in self._ops:
            vp = v.extract(op.domain)
            lp = MultiLinearization.make_var(vp, x.maxderiv)
            rl = op.apply(lp)
            res = res + rl.val
            pjacs.append(rl.jacs)
            j0 = rl.jacs[0] if j0 is None else j0 + rl.jacs[0]
        jacs = [j0, ]
        for i in range(1, x.maxderiv):
            tm = [jacj[i] for jacj in pjacs if jacj[i] is not None]
            jacs.append(None if len(tm) == 0 else SumTensor(tm))
        return x.new(res, jacs)

class MultiPointwiseProduct(MultiOperator):
    def __init__(self, op1, op2):
        self._target = op1.target
        assert op1.target == op2.target
        self._domain = ift.domain_union([op1.domain, op2.domain])
        self._op1, self._op2 = op1, op2

    def apply(self, x):
        v = x.val
        v1 = v.extract(self._op1.domain)
        vl1 = MultiLinearization.make_var(v1, x.maxderiv)
        rl1 = self._op1(vl1)
        v2 = v.extract(self._op2.domain)
        vl2 = MultiLinearization.make_var(v2, x.maxderiv)
        rl2 = self._op2(vl2)
        res = rl1.val*rl2.val
        jacs = [ift.makeOp(rl2.val)@rl1.jacs[0] + ift.makeOp(rl1.val)@rl2.jacs[0], ]
        for i in range(x.maxderiv)[1:]:
            jacs.append(GenLeibnizTensor(rl1.jacs[:(i+1)], rl2.jacs[:(i+1)],
                                         rl1.val, rl2.val))
        return x.new(res, jacs)

class MultiOpChain(MultiOperator):
    def __init__(self, oplist):
        self._oplist = oplist
        self._domain = oplist[0].domain
        self._target = oplist[-1].target
        for i in range(len(oplist)-1):
            assert oplist[i].target == oplist[i+1].domain

    def apply(self, x):
        for op in self._oplist:
            x = op.apply(x)
        return x
