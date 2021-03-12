import jifty1 as jft
from jax import value_and_grad
import jax.numpy as np
from numpy import testing

def test_ncg():
    pos = [0., (3.,),  {"a":5.}]
    getters = [lambda x: x[0],
            lambda x: x[1][0],
            lambda x: x[2]["a"]]
    tgt = [-10.,1.,2.]
    met = [10.,40.,2]
    def model(p):
        losses = []
        for i, get in enumerate(getters):
            losses.append((get(p)-tgt[i])**2*met[i])
        return np.sum(np.array(losses))
    def metric(p, tan):
        return [tan[0]*met[0],
                (tan[1][0]*met[1],),
                {"a":tan[2]["a"]*met[2]}]
    res = jft.NCG(pos,
            value_and_grad(model),
            metric,
            iterations=10,
            absdelta=1e-6)
    for i,get in enumerate(getters):
        testing.assert_allclose(get(res), tgt[i], atol=1e-6,rtol=1e-5)

if __name__=="__main__":
    test_ncg()

