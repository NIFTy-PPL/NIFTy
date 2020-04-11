import nifty6 as ift
from differential_tensor import DiagonalTensor, SumTensor
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
        for i in range(len(self._jacs))[1:]:
            jacs.append(DiagonalTensor(self._domain, self._target,
                                       self._jacs[i](v), i+1))
        return x.new(res, jacs)

class MultiLocalExp(MultiOperator):
    def __init__(self, domain):
        self._domain = ift.makeDomain(domain)
        self._target = ift.makeDomain(domain)

    def apply(self, x):
        v = x.val
        res = v.exp()
        jacs = [ift.makeOp(res), ]
        for i in range(x.maxderiv)[1:]:
            jacs.append(DiagonalTensor(self._domain, self._target,
                                       res, i+1))
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
            if j0 == None:
                j0 = rl.jacs[0]
            else:
                j0 = j0 + rl.jacs[0]
        jacs = [j0, ]
        for i in range(x.maxderiv)[1:]:
            tm = []
            for j in range(len(pjacs)):
                if pjacs[j][i] is not None:
                    tm.append(pjacs[j][i])
            if len(tm) != 0:
                jacs.append(SumTensor(tm))
            else:
                jacs.append(None)
        return x.new(res, jacs)

class MultiOpChain(MultiOperator):
    def __init__(self, oplist):
        self._oplist = oplist
        self._domain = oplist[0].domain
        self._target = oplist[-1].target
        for i in range(len(oplist)-1):
            assert oplist[i].target == oplist[i+1].domain

    def apply(self, x):
        res = self._oplist[0].apply(x)
        for i in range(len(self._oplist)-1):
            res = self._oplist[i+1].apply(res)
        return res
