from jax import jvp, vjp
from jax.scipy.sparse.linalg import cg #TODO: replace
from itertools import repeat


from .sugar import just_add, makeField


class Likelihood():
    def __init__(
        self,
        energy,
        metric,
        draw_metric_sample=None,
    ):
        self._hamiltonian = energy
        self._metric = metric
        self._draw_metric_sample = draw_metric_sample

    def __call__(self, primals):
        return self._hamiltonian(primals)

    def __matmul__(self, f):
        nham = lambda x: self.energy(f(x))
        def met(primals, tangents):
            y, t = jvp(f, (primals,), (tangents,))
            r = self.metric(y, t)
            _, bwd = vjp(f, primals)
            res = bwd(r)
            return res[0]

        def draw_sample(primals, key):
            y, bwd = vjp(f, primals)
            samp, nkey = self.draw_sample(y, key)
            return bwd(samp)[0], nkey

        return Likelihood(nham, met, draw_sample)

    def __add__(self, ham):
        if not isinstance(ham, Likelihood):
            te = (
                "object which to add to this instance is of invalid type"
                f" {type(ham)!r}"
            )
            raise TypeError(te)

        def draw_metric_sample(primals, key, **kwargs):
            # Ensure that samples are not inverted in any of the recursive calls
            assert "from_inverse" not in kwargs
            # Ensure there is no prior for the CG algorithm in recursive calls
            # as the prior is sensible only for the top-level likelihood
            assert "x0" not in kwargs

            key, subkeys = random.split(key, 2)
            smpl_self, _ = self.draw_sample(primals, key=subkeys[0], **kwargs)
            smpl_other, _ = ham.draw_sample(primals, key=subkeys[1], **kwargs)

            return just_add(smpl_self, smpl_other), key
        
        return Likelihood(
            energy=lambda p: self(p) + ham(p),
            metric=lambda p, t: just_add(self.metric(p, t), ham.metric(p, t)),
            draw_metric_sample=draw_metric_sample
        )

    def energy(self, primals):
        return self._hamiltonian(primals)

    def metric(self, primals, tangents):
        return self._metric(primals, tangents)

    def draw_sample(
        self,
        primals,
        key,
        from_inverse = False,
        x0 = None,
        maxiter = None,
        **kwargs
    ):
        if not self._draw_metric_sample:
            nie = "`draw_sample` is not implemented"
            raise NotImplementedError(nie)

        if from_inverse:
            nie = "Cannot draw from the inverse of this operator"
            raise NotImplementedError(nie)
        else:
            return self._draw_metric_sample(primals, key=key, **kwargs)

