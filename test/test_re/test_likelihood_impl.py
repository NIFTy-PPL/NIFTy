#!/usr/bin/env python3

import operator
from functools import partial, reduce

import jax
import jax.numpy as jnp
import nifty.re as jft
import numpy as np
import pytest
from jax import random
from jax.tree_util import tree_map
from numpy.testing import assert_allclose

jax.config.update("jax_enable_x64", True)

pmp = pytest.mark.parametrize


class _OverwriteableLikelihood(jft.Likelihood):
    def __init__(
        self,
        energy=None,
        *,
        metric=None,
        left_sqrt_metric=None,
        right_sqrt_metric=None,
        transformation=None,
        domain=None,
        lsm_tangents_shape=None,
    ):
        self._energy = energy
        self._metric = metric
        self._left_sqrt_metric = left_sqrt_metric
        self._right_sqrt_metric = right_sqrt_metric
        self._transformation = transformation
        super().__init__(domain=domain, lsm_tangents_shape=lsm_tangents_shape)

    def energy(self, *a, **k):
        if self._energy is not None:
            return self._energy(*a, **k)
        return super().energy(*a, **k)

    def metric(self, *a, **k):
        if self._metric is not None:
            return self._metric(*a, **k)
        return super().metric(*a, **k)

    def left_sqrt_metric(self, *a, **k):
        if self._left_sqrt_metric is not None:
            return self._left_sqrt_metric(*a, **k)
        return super().left_sqrt_metric(*a, **k)

    def right_sqrt_metric(self, *a, **k):
        if self._right_sqrt_metric is not None:
            return self._right_sqrt_metric(*a, **k)
        return super().right_sqrt_metric(*a, **k)

    def transformation(self, *a, **k):
        if self._transformation is not None:
            return self._transformation(*a, **k)
        return super().transformation(*a, **k)


def lst2fixt(lst):
    @pytest.fixture(params=lst)
    def fixt(request):
        return request.param

    return fixt


def random_draw(key, shape, dtype, method):
    def _isleaf(x):
        if isinstance(x, tuple):
            return bool(reduce(operator.mul, (isinstance(ii, int) for ii in x)))
        return False

    swd = tree_map(lambda x: jft.ShapeWithDtype(x, dtype), shape, is_leaf=_isleaf)
    return jft.random_like(key, jft.Vector(swd), method)


def expand_shape(shape1, shape2):
    def _isleaf(x):
        if isinstance(x, tuple):
            return bool(reduce(operator.mul, (isinstance(ii, int) for ii in x)))
        return False

    return tree_map(lambda x: x + shape2, shape1, is_leaf=_isleaf)


def random_draw_cov(key, shape):
    @partial(jnp.vectorize, signature="(),(),()->(2,2)")
    def build_cov(ev1, ev2, angle):
        rot = jnp.array(
            [[jnp.cos(angle), -jnp.sin(angle)], [jnp.sin(angle), jnp.cos(angle)]]
        )
        cov_diag = jnp.array([[ev1, 0.0], [0.0, ev2]])
        return rot @ cov_diag @ rot.T

    sk = list(random.split(key, 3))
    ev1 = 1.0e-5 + random_draw(sk.pop(), shape, float, random.exponential)
    ev2 = ev1 + 1.0e-5 + random_draw(sk.pop(), shape, float, random.exponential)
    angle = random_draw(sk.pop(), shape, float, random.normal)
    return tree_map(build_cov, ev1, ev2, angle)


def random_noise_std_inv(key, shape):
    diag = 1.0 / random_draw(key, shape, float, random.exponential)

    def noise_std_inv(tangents):
        return diag * tangents

    return noise_std_inv


seed = lst2fixt((3639, 12, 41, 42))
shape = lst2fixt(((4, 2), (2, 1), (5,), [(2, 3), (1, 2)], ((2,), {"a": (3, 1)})))

lh_init_true = (
    (
        jft.Gaussian,
        {
            "data": partial(random_draw, dtype=float, method=random.normal),
            "noise_std_inv": random_noise_std_inv,
        },
        partial(random_draw, dtype=float, method=random.normal),
    ),
    (
        jft.StudentT,
        {
            "data": partial(random_draw, dtype=float, method=random.normal),
            "dof": partial(random_draw, dtype=float, method=random.exponential),
            "noise_std_inv": random_noise_std_inv,
        },
        partial(random_draw, dtype=float, method=random.normal),
    ),
    (
        jft.Poissonian,
        {
            "data": partial(
                random_draw, dtype=int, method=partial(random.poisson, lam=3.14)
            ),
        },
        partial(random_draw, dtype=float, method=random.exponential),
    ),
)
lh_init_approx = (
    (
        jft.VariableCovarianceGaussian,
        {
            "data": partial(random_draw, dtype=float, method=random.normal),
        },
        lambda key, shape: (
            random_draw(key, shape, float, random.normal),
            3.0
            + 1.0 / tree_map(jnp.exp, random_draw(key, shape, float, random.normal)),
        ),
    ),
    (
        jft.VariableCovarianceStudentT,
        {
            "data": partial(random_draw, dtype=float, method=random.normal),
            "dof": partial(random_draw, dtype=float, method=random.exponential),
        },
        lambda key, shape: (
            random_draw(key, shape, float, random.normal),
            tree_map(
                jnp.exp, 3.0 + 1e-1 * random_draw(key, shape, float, random.normal)
            ),
        ),
    ),
    (
        jft.NDVariableCovarianceGaussian,
        {
            "data": lambda key, shape: (
                partial(random_draw, dtype=float, method=random.normal)(
                    key, expand_shape(shape, (2,))
                )
            ),
        },
        lambda key, shape: (
            random_draw(key, expand_shape(shape, (2,)), float, random.normal),
            random_draw_cov(key, shape),
        ),
    ),
)


def test_gaussian_vs_vcgaussian_consistency(seed, shape):
    rtol = 10 * jnp.finfo(jnp.zeros(0).dtype).eps
    atol = 10 * jnp.finfo(jnp.zeros(0).dtype).eps

    key = random.PRNGKey(seed)
    sk = list(random.split(key, 5))
    d = random_draw(sk.pop(), shape, float, random.normal)
    m1 = random_draw(sk.pop(), shape, float, random.normal)
    m2 = random_draw(sk.pop(), shape, float, random.normal)
    t = random_draw(sk.pop(), shape, float, random.normal)
    inv_std = random_draw(sk.pop(), shape, float, random.normal)
    inv_std = 1.0 / tree_map(jnp.exp, 1.0 + inv_std)

    gauss = jft.Gaussian(d, noise_std_inv=lambda x: inv_std * x)
    vcgauss = jft.VariableCovarianceGaussian(d)

    diff_g = gauss(m2) - gauss(m1)
    diff_vcg = vcgauss((m2, inv_std)) - vcgauss((m1, inv_std))

    tree_map(partial(assert_allclose, rtol=rtol, atol=atol), diff_g, diff_vcg)

    met_g = gauss.metric(m1, t)
    met_vcg = vcgauss.metric((m1, inv_std), (t, d / 2))[0]
    tree_map(partial(assert_allclose, rtol=rtol, atol=atol), met_g, met_vcg)


def test_studt_vs_vcstudt_consistency(seed, shape):
    rtol = 10 * jnp.finfo(jnp.zeros(0).dtype).eps
    atol = 4 * jnp.finfo(jnp.zeros(0).dtype).eps

    key = random.PRNGKey(seed)
    sk = list(random.split(key, 6))
    d = random_draw(sk.pop(), shape, float, random.normal)
    dof = random_draw(sk.pop(), shape, float, random.normal)
    m1 = random_draw(sk.pop(), shape, float, random.normal)
    m2 = random_draw(sk.pop(), shape, float, random.normal)
    t = random_draw(sk.pop(), shape, float, random.normal)
    inv_std = random_draw(sk.pop(), shape, float, random.normal)
    inv_std = 1.0 / tree_map(jnp.exp, 1.0 + inv_std)

    studt = jft.StudentT(d, dof, noise_std_inv=lambda x: inv_std * x)
    vcstudt = jft.VariableCovarianceStudentT(d, dof)

    diff_t = studt(m2) - studt(m1)
    diff_vct = vcstudt((m2, 1.0 / inv_std)) - vcstudt((m1, 1.0 / inv_std))
    tree_map(partial(assert_allclose, rtol=rtol, atol=atol), diff_t, diff_vct)

    met_t = studt.metric(m1, t)
    met_vct = vcstudt.metric((m1, 1.0 / inv_std), (t, d / 2))[0]
    tree_map(partial(assert_allclose, rtol=rtol, atol=atol), met_t, met_vct)


@pmp("lh_init", lh_init_true + lh_init_approx)
def test_sqrt_metric_vs_metric_consistency(seed, shape, lh_init):
    rtol = 10 * jnp.finfo(jnp.zeros(0).dtype).eps
    atol = 1.0e-8
    aallclose = partial(assert_allclose, rtol=rtol, atol=atol)

    N_TRIES = 5

    lh_init_method, draw, latent_init = lh_init
    key = random.PRNGKey(seed)
    key, *subkeys = random.split(key, 1 + len(draw))
    init_kwargs = {
        k: method(key=sk, shape=shape) for (k, method), sk in zip(draw.items(), subkeys)
    }
    lh = lh_init_method(**init_kwargs)

    # Let NIFTy.re infer the metric from the left-square-root-metric
    lh_mini1 = _OverwriteableLikelihood(
        lh.energy,
        left_sqrt_metric=lh.left_sqrt_metric,
        lsm_tangents_shape=lh.lsm_tangents_shape,
    )

    for _ in range(N_TRIES):
        key, *sk = random.split(key, 3)
        p = latent_init(sk.pop(), shape=shape)
        t = latent_init(sk.pop(), shape=shape)
        tree_map(aallclose, lh.metric(p, t), lh_mini1.metric(p, t))

    # Let NIFTy.re infer the metric from the left-square-root-metric and
    # right-sqrt-metric
    lh_mini2 = _OverwriteableLikelihood(
        lh.energy,
        left_sqrt_metric=lh.left_sqrt_metric,
        right_sqrt_metric=lh.right_sqrt_metric,
        lsm_tangents_shape=lh.lsm_tangents_shape,
    )

    for _ in range(N_TRIES):
        key, *sk = random.split(key, 3)
        p = latent_init(sk.pop(), shape=shape)
        t = latent_init(sk.pop(), shape=shape)
        tree_map(aallclose, lh.metric(p, t), lh_mini2.metric(p, t))


@pmp("lh_init", lh_init_true)
def test_transformation_vs_sqrt_metric_consistency(seed, shape, lh_init):
    rtol = 4 * jnp.finfo(jnp.zeros(0).dtype).eps
    atol = 0.0
    aallclose = partial(assert_allclose, rtol=rtol, atol=atol)

    N_TRIES = 5

    lh_init_method, draw, latent_init = lh_init
    key = random.PRNGKey(seed)
    key, *subkeys = random.split(key, 1 + len(draw))
    init_kwargs = {
        k: method(key=sk, shape=shape) for (k, method), sk in zip(draw.items(), subkeys)
    }
    lh = lh_init_method(**init_kwargs)
    try:
        lh.transformation(latent_init(key, shape=shape))
    except NotImplementedError:
        pytest.skip("no transformation rule implemented yet")

    energy, lsm, lsm_shp = lh.energy, lh.left_sqrt_metric, lh.lsm_tangents_shape
    # Let NIFTy.re infer the left-square-root-metric and the metric from the
    # transformation
    lh_mini = _OverwriteableLikelihood(
        energy, left_sqrt_metric=lsm, lsm_tangents_shape=lsm_shp
    )

    for _ in range(N_TRIES):
        key, *sk = random.split(key, 3)
        p = latent_init(sk.pop(), shape=shape)
        t = latent_init(sk.pop(), shape=shape)
        tree_map(aallclose, lh.left_sqrt_metric(p, t), lh_mini.left_sqrt_metric(p, t))
        tree_map(aallclose, lh.right_sqrt_metric(p, t), lh_mini.right_sqrt_metric(p, t))
        tree_map(aallclose, lh.metric(p, t), lh_mini.metric(p, t))


@pmp("lh_init", lh_init_true + lh_init_approx)
def test_residuals_allfinite(seed, shape, lh_init):
    allfinite = lambda x: jnp.all(jnp.isfinite(x))

    N_TRIES = 5

    lh_init_method, draw, latent_init = lh_init
    key = random.PRNGKey(seed)
    key, *subkeys = random.split(key, 1 + len(draw))
    init_kwargs = {
        k: method(key=sk, shape=shape) for (k, method), sk in zip(draw.items(), subkeys)
    }
    lh = lh_init_method(**init_kwargs)

    residual = lh.normalized_residual
    for _ in range(N_TRIES):
        key, sk = random.split(key, 2)
        r = residual(latent_init(sk, shape=shape))
        assert jax.tree_util.tree_reduce(lambda a, b: a * b, tree_map(allfinite, r))


@pmp("iscomplex", [False, True])
def test_nifty_vcgaussian_vs_niftyre_vcgaussian_consistency(seed, iscomplex):
    import nifty.cl as ift

    key = random.PRNGKey(seed)

    sp = ift.RGSpace(1)
    if iscomplex:
        val = 1.0 + 1.0j
        data = ift.full(sp, 0.0 + 0.0j)
    else:
        val = 1.0
        data = ift.full(sp, 0.0)
    fl = ift.full(sp, val)
    rls = ift.Realizer(sp)
    res = fl * rls.adjoint - data
    res = res.ducktape("res")
    res = res.ducktape_left("res")
    invcov = ift.exp(ift.makeOp(ift.full(sp, 1.0)))
    invcov = invcov.ducktape("invcov")
    invcov = invcov.ducktape_left("invcov")
    op = res + invcov
    dt = jnp.complex128 if iscomplex else jnp.float64
    varcov_nft = ift.VariableCovarianceGaussianEnergy(sp, "res", "invcov", dt) @ op

    def op_jft(x):
        return [val * x["res"], jnp.sqrt(jnp.exp(x["invcov"]))]

    varcov_jft = jft.VariableCovarianceGaussian(data.asnumpy()).amend(op_jft)

    def shapewithdtype_from_domain(domain, dtype=float):
        if isinstance(dtype, dict):
            dtp_fallback = float  # Fallback to `float` for unspecified keys
            k2dtp = dtype
        else:
            dtp_fallback = dtype
            k2dtp = {}

        if isinstance(domain, ift.MultiDomain):
            parameter_tree = {}
            for k, dom in domain.items():
                parameter_tree[k] = jft.ShapeWithDtype(
                    dom.shape, k2dtp.get(k, dtp_fallback)
                )
        elif isinstance(domain, ift.DomainTuple):
            parameter_tree = jft.ShapeWithDtype(domain.shape, dtype)
        else:
            raise TypeError(f"incompatible domain {domain!r}")
        return parameter_tree

    def random_jft_ift_field(key):
        inp = jft.random_like(key, shapewithdtype_from_domain(op.domain))
        inp_nft = ift.MultiField(
            op.domain, tree_map(
                ift.Field,
                op.domain.values(),
                tuple(np.array(v) for v in inp.values()),
            )
        )
        return inp, inp_nft

    # Test value
    key, sk = random.split(key)
    inp_jft, inp_nft = random_jft_ift_field(sk)
    lh_jft = varcov_jft(inp_jft)
    lh_nft = varcov_nft(inp_nft)
    print(f"test value :: nifty: {lh_nft.asnumpy():.4f} :: jft: {lh_jft:.4f}")
    assert_allclose(lh_nft.asnumpy(), lh_jft)

    # Test metric
    key, sk = random.split(key)
    inp2_jft, inp2_nft = random_jft_ift_field(sk)
    lin = varcov_nft(ift.Linearization.make_var(inp_nft, want_metric=True))
    met_nft = lin.metric
    met_res_nft = met_nft(inp2_nft)
    met_res_jft = varcov_jft.metric(inp_jft, inp2_jft)
    print(f"test metric ::\nnifty: {met_res_nft.val}\njft: {met_res_jft}\n")
    assert_allclose(met_res_nft["invcov"].asnumpy(), met_res_jft["invcov"])
    assert_allclose(met_res_nft["res"].asnumpy(), met_res_jft["res"])

    # Test transform
    key, sk = random.split(key)
    inp3_jft, inp3_nft = random_jft_ift_field(sk)
    trafo_nft = varcov_nft.get_transformation()[1](inp3_nft)
    trafo_jft = varcov_jft.transformation(inp3_jft)
    print(f"test transform:\nnifty: {trafo_nft.val}\njft: {trafo_jft}\n")
    assert_allclose(trafo_nft["res"].asnumpy(), trafo_jft[0])
    assert_allclose(trafo_nft["invcov"].asnumpy(), trafo_jft[1])


@pmp("lh_init", lh_init_true)
def test_transformation_metric_consistency(seed, shape, lh_init):
    rtol = 10 * jnp.finfo(jnp.zeros(0).dtype).eps
    atol = 0.0
    aallclose = partial(assert_allclose, rtol=rtol, atol=atol)

    N_TRIES = 5

    lh_init_method, draw, latent_init = lh_init
    key = random.PRNGKey(seed)
    key, *subkeys = random.split(key, 1 + len(draw))
    init_kwargs = {
        k: method(key=sk, shape=shape) for (k, method), sk in zip(draw.items(), subkeys)
    }
    lh = lh_init_method(**init_kwargs)
    try:
        lh.transformation(latent_init(key, shape=shape))
    except NotImplementedError:
        pytest.skip("no transformation rule implemented yet")

    def gram_jacobian_matrix(f, x):
        """Computes J[f](x)^T @ J[f](x) as one large matrix."""
        x_flat, unravel = jax.flatten_util.ravel_pytree(x)

        def f_ravelled(z_flat):
            z = unravel(z_flat)
            res_flat, _ = jax.flatten_util.ravel_pytree(f(z))
            return res_flat

        J = jax.jacfwd(f_ravelled)(x_flat)
        return J.T @ J

    def linop_to_matrix(f, example_input):
        """recovers the matrix representing a linear function f
        given the pytree of an example input to f"""
        ex_flat, unravel = jax.flatten_util.ravel_pytree(example_input)

        def f_ravelled(x):
            x_flat = unravel(x)
            y_flat, _ = jax.flatten_util.ravel_pytree(f(x_flat))
            return y_flat

        basis_vectors = jnp.eye(len(ex_flat))
        return jax.vmap(f_ravelled)(basis_vectors).T

    for _ in range(N_TRIES):
        key, subkey = random.split(key, 2)
        primals = latent_init(subkey, shape=shape)
        fm = linop_to_matrix(lambda x: lh.metric(primals, x), jft.zeros_like(primals))
        gm = gram_jacobian_matrix(lh.transformation, primals)
        aallclose(fm, gm)


if __name__ == "__main__":
    test_gaussian_vs_vcgaussian_consistency(42, (5,))
