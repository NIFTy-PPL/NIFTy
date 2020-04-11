import nifty6 as ift
import numpy as np
from multi_linearization import MultiLinearization
from multiderivative_operator import (MultiOpChain, MultiLocalNonlin,
                                      MultiLocalExp, MultiLinearOperator,
                                      MultiSumOperator)

def exp(x):
    return ift.exp(2.*x)
def df(x):
    return 2.*ift.exp(2.*x)
def ddf(x):
    return 4.*ift.exp(2.*x)
expjacs = [df, ddf]

dom = ift.RGSpace(256)

x = ift.from_random('normal', dom)
ml = MultiLinearization.make_var(x, 2)
op = MultiLocalNonlin(dom, exp, expjacs)

res = op.apply(ml)
dx1 = ift.from_random('normal', dom)
dx2 = ift.from_random('normal', dom)
grlist = [dx1, dx2]

myjacs = []
myjacs.append(res.jacs[0](grlist[0]))
myjacs.append(res.jacs[1].apply(grlist))

ex = np.exp(2.*x.val)
assert np.allclose(res.val.val, ex)
assert np.allclose(myjacs[0].val, 2.*ex*dx1.val)
assert np.allclose(myjacs[1].val, 4.*ex*dx1.val*dx2.val)

res = MultiOpChain([op,op]).apply(ml)
myjacs = []
myjacs.append(res.jacs[0](grlist[0]))
myjacs.append(res.jacs[1].apply(grlist))

ex = np.exp(2.*x.val)
exex = np.exp(2.*ex)
assert np.allclose(res.val.val, exex)
assert np.allclose(myjacs[0].val, exex*2.*ex*2.*dx1.val)
r = exex*(2.*ex*2.)**2 * dx1.val*dx2.val
r += exex*2. * ex*4.*dx1.val*dx2.val
assert np.allclose(myjacs[1].val, r)

conop = res.jacs[1].contract_to_one([dx1, ])
r2 = conop(dx2)
assert np.allclose(r2.val, myjacs[1].val)

op0 = ift.FieldAdapter(dom, 'key')
op0 = MultiLinearOperator(op0)
op = MultiLocalExp(dom)
op2 = MultiLinearOperator(ift.FFTOperator(dom))

x = ift.from_random('normal', op0._domain)
ml = MultiLinearization.make_var(x, 3)
res = MultiOpChain([op0,op,op2]).apply(ml)

dxs = [ift.from_random('normal', x.domain),]
myres = [res.jacs[0](dxs[0]), ]
cons = []
for i in range(len(res.jacs))[1:]:
    dxs.append(ift.from_random('normal', x.domain))
    myres.append(res.jacs[i].apply(dxs))
    tm = res.jacs[i].contract_to_one(dxs[:-1])
    cons.append(tm)
    assert np.allclose(myres[-1].val, (tm(dxs[-1])).val)




fa = ift.FieldAdapter(dom, "key0")
fa2 = ift.FieldAdapter(dom, "key1")

fa = MultiLinearOperator(fa)
fa = MultiOpChain([fa, MultiLocalExp(dom)])
fa2 = MultiLinearOperator(fa2)

op = MultiSumOperator([fa,fa2])

x = ift.from_random('normal', op.domain)
xl = MultiLinearization.make_var(x, 4)

res = op.apply(xl)

dxs = [ift.from_random('normal', x.domain),]
myres = [res.jacs[0](dxs[0]), ]
cons = []
for i in range(len(res.jacs))[1:]:
    dxs.append(ift.from_random('normal', x.domain))
    myres.append(res.jacs[i].apply(dxs))
    tm = res.jacs[i].contract_to_one(dxs[:-1])
    cons.append(tm)
    assert np.allclose(myres[-1].val, (tm(dxs[-1])).val)