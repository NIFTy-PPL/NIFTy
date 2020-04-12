import nifty6 as ift
import numpy as np
from multi_linearization import MultiLinearization
from multiderivative_operator import (MultiOpChain, MultiLocalNonlin,
                                      MultiLocalExp, MultiLinearOperator,
                                      MultiSumOperator, MultiPointwiseProduct)
import random

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

res = op(ml)
dx1 = ift.from_random('normal', dom)
dx2 = ift.from_random('normal', dom)
grlist = [dx1, dx2]

myjacs = []
myjacs.append(res.jacs[0](grlist[0]))
myjacs.append(res.jacs[1](grlist))

ex = np.exp(2.*x.val)
assert np.allclose(res.val.val, ex)
assert np.allclose(myjacs[0].val, 2.*ex*dx1.val)
assert np.allclose(myjacs[1].val, 4.*ex*dx1.val*dx2.val)

res = MultiOpChain([op,op])(ml)
myjacs = []
myjacs.append(res.jacs[0](grlist[0]))
myjacs.append(res.jacs[1](grlist))

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
ml = MultiLinearization.make_var(x, 4)
res = MultiOpChain([op0,op,op2])(ml)

dxs = [ift.from_random('normal', x.domain),]
myres = [res.jacs[0](dxs[0]), ]
cons = []
for i in range(len(res.jacs))[1:]:
    dxs.append(ift.from_random('normal', x.domain))
    myres.append(res.jacs[i](dxs))
    tm = res.jacs[i].contract_to_one(dxs[:-1])
    cons.append(tm)
    assert np.allclose(myres[-1].val, (tm(dxs[-1])).val)

tm = res.jacs[-1].partial_contract(dxs[:2])
re = tm(dxs[2:])
assert np.allclose(myres[-1].val, re.val)

random.shuffle(dxs)
r2 = res.jacs[-1](dxs)
assert np.allclose(myres[-1].val, r2.val)


fa = ift.FieldAdapter(dom, "key0")
fa2 = ift.FieldAdapter(dom, "key1")

fa = MultiLinearOperator(fa)
fa = MultiOpChain([fa, MultiLocalExp(dom)])
fa2 = MultiLinearOperator(fa2)

op = MultiSumOperator([fa,fa2])

x = ift.from_random('normal', op.domain)
xl = MultiLinearization.make_var(x, 4)

res = op(xl)

xl1 = MultiLinearization.make_var(x.extract(fa.domain), 4)
r1 = fa(xl1)
xl2 = MultiLinearization.make_var(x.extract(fa2.domain), 4)
r2 = fa2(xl2)

dxs = [ift.from_random('normal', x.domain),]
myres = [res.jacs[0](dxs[0]), ]
myres2 = [r1.jacs[0](dxs[0].extract(r1.jacs[0].domain)) + 
          r2.jacs[0](dxs[0].extract(r2.jacs[0].domain))]
cons = []
for i in range(len(res.jacs))[1:]:
    dxs.append(ift.from_random('normal', x.domain))
    myres.append(res.jacs[i](dxs))
    dxs1, dxs2 = [], []
    for j in range(len(dxs)):
        dxs1.append(dxs[j].extract(r1.jacs[i].domain))
        dxs2.append(dxs[j].extract(r2.jacs[i].domain))
    myres2.append(r1.jacs[i](dxs1)+r2.jacs[i](dxs2))
    tm = res.jacs[i].contract_to_one(dxs[:-1])
    cons.append(tm)
    assert np.allclose(myres[-1].val, (tm(dxs[-1])).val)

for i in range(len(myres)):
    assert np.allclose(myres[i].val, myres2[i].val)


no = 4
op = MultiPointwiseProduct(fa,fa2)
x = ift.from_random('normal', op.domain)
xl = MultiLinearization.make_var(x, no)
rl = op(xl)

xl1 = MultiLinearization.make_var(x.extract(fa.domain), no)
r1 = fa(xl1)
xl2 = MultiLinearization.make_var(x.extract(fa2.domain), no)
r2 = fa2(xl2)

dxs = [ift.from_random('normal', xl.domain) for _ in range(no)]
dxs1, dxs2 = [], []
for j in range(len(dxs)):
    dxs1.append(dxs[j].extract(fa.domain))
    dxs2.append(dxs[j].extract(fa2.domain))

myrs = [rl.jacs[0](dxs[0]), ]
for i in range(no)[1:]:
    myrs.append(rl.jacs[i](dxs[:(i+1)]))

twotest = r1.jacs[1](dxs1[:2])*r2.val
twotest = twotest + r1.jacs[0](dxs1[0])*r2.jacs[0](dxs2[1])
twotest = twotest + r1.jacs[0](dxs1[1])*r2.jacs[0](dxs2[0])
twotest = twotest + r1.val*r2.jacs[1](dxs2[:2])
assert np.allclose(twotest.val, myrs[1].val)

random.shuffle(dxs)
r2 = rl.jacs[0](dxs[0])
assert np.allclose(myrs[0].val, r2.val)