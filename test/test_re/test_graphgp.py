import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import graphgp as gp
import nifty.re as jft


jax.config.update("jax_enable_x64", True)


def _setup(n_partitions=None, output_shape=(32,)):
    n_partitions = min(jax.device_count(), 4) if n_partitions is None else n_partitions
    n = int(np.prod(output_shape))
    points = jnp.linspace(0.0, 1.0, n, dtype=jnp.float64)[:, None]
    graph = gp.build_graph(points, n0=8, k=4)
    covariance = gp.extras.matern_kernel(
        p=0,
        variance=1.0,
        cutoff=0.3,
        r_min=1e-4,
        r_max=2.0,
        n_bins=64,
        jitter=1e-5,
    )
    owners = np.arange(n, dtype=np.int32) % n_partitions
    plan = gp.distributed.partition_graph(
        graph, owners, output_shape=output_shape
    )
    mesh = Mesh(np.asarray(jax.devices()[:n_partitions]), ("space",))
    return graph, covariance, plan, mesh


def test_fixed_covariance_domain_initializer_prefix_and_offset():
    graph, covariance, plan, mesh = _setup(n_partitions=1, output_shape=(8, 4))
    model = jft.GraphGPField(
        plan, covariance, mesh=mesh, prefix="dust_", offset=1.25
    )
    position = model.init(jax.random.key(0))
    actual = model(position)
    expected = (
        gp.generate(graph, covariance, position["dust_excitations"].reshape(-1))
        .reshape(plan.output_shape)
        + 1.25
    )

    assert model.excitation_name == "dust_excitations"
    assert model.domain[model.excitation_name].shape == (8, 4)
    assert model.target.shape == (8, 4)
    assert actual.shape == model.target.shape
    assert jnp.allclose(actual, expected, rtol=1e-10, atol=1e-10)
    assert position[model.excitation_name].sharding.spec == P("space", None)


def test_modeled_covariance_jit_autodiff_and_metric():
    _, _, plan, mesh = _setup(n_partitions=1)
    covariance = jft.MaternCovarianceModel(
        Ndim=1,
        r_min=1e-4,
        r_max=2.0,
        variance=(1.0, 0.1),
        lengthscale=(0.3, 0.05),
        negloglogslope=3.0,
        Ninterp=64,
        mode="graphgp",
        prefix="cov_",
    )
    offset = jft.NormalPrior(0.0, 0.1, name="field_offset")
    field = jft.GraphGPField(
        plan, covariance, mesh=mesh, prefix="gas_", offset=offset
    )
    position = field.init(jax.random.key(1))
    tangent = jft.random_like(jax.random.key(2), position)

    actual = jax.jit(field)(position)
    _, jvp = jax.jvp(field, (position,), (tangent,))
    _, pullback = jax.vjp(field, position)
    vjp = pullback(jnp.ones(field.target.shape))[0]
    grad = jax.grad(lambda x: jnp.sum(field(x) ** 2))(position)

    assert jnp.all(jnp.isfinite(actual))
    assert jnp.all(jnp.isfinite(jvp))
    assert all(jnp.all(jnp.isfinite(v)) for v in jax.tree.leaves(vjp))
    assert all(jnp.all(jnp.isfinite(v)) for v in jax.tree.leaves(grad))

    signal = jft.Model(
        lambda x: jnp.tanh(field(x)),
        domain=field.domain,
        init=field.init,
    )
    likelihood = jft.Gaussian(
        data=jnp.zeros(field.target.shape), noise_cov_inv=lambda x: 4.0 * x
    ).amend(signal)
    metric = likelihood.metric(position, tangent)
    assert all(jnp.all(jnp.isfinite(v)) for v in jax.tree.leaves(metric))


def test_missing_dependency_message(monkeypatch):
    import nifty.re.graphgp as adapter

    _, covariance, plan, mesh = _setup(n_partitions=1)
    real_import = adapter.import_module

    def missing(name):
        if name == "graphgp":
            raise ImportError("not installed")
        return real_import(name)

    monkeypatch.setattr(adapter, "import_module", missing)
    with pytest.raises(ImportError, match=r"nifty\[re_graphgp\]"):
        jft.GraphGPField(plan, covariance, mesh=mesh)


@pytest.mark.skipif(jax.device_count() < 4, reason="requires four devices")
def test_four_device_spatial_and_two_by_two_sample_space_meshes():
    graph, covariance, plan4, mesh4 = _setup(n_partitions=4)
    field4 = jft.GraphGPField(plan4, covariance, mesh=mesh4)
    position4 = field4.init(jax.random.key(3))
    actual = jax.jit(field4)(position4)
    expected = gp.generate(
        graph, covariance, position4[field4.excitation_name].reshape(-1)
    )
    assert jnp.allclose(actual, expected, rtol=1e-10, atol=1e-10)
    assert actual.sharding.spec == P("space")

    _, _, plan2, _ = _setup(n_partitions=2)
    mesh2d = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("sample", "space"))
    field2 = jft.GraphGPField(plan2, covariance, mesh=mesh2d)
    position = field2.init(jax.random.key(4))
    layout = field2.sharding_layout(sample_axis="sample")
    samples = jax.tree.map(lambda x: jnp.stack((x, -x)), position)
    samples = jax.device_put(samples, layout.sample_shardings(position))
    mapped = jax.jit(
        jax.vmap(field2),
        in_shardings=(layout.sample_shardings(position),),
        out_shardings=NamedSharding(mesh2d, P("sample", "space")),
    )(samples)

    assert mapped.shape == (2,) + field2.target.shape
    assert mapped.sharding.spec == P("sample", "space")
    assert layout.position_specs_for({"other": jnp.array(1.0)})["other"] == P()


def test_devices_and_sharding_are_mutually_exclusive():
    _, _, plan, mesh = _setup(n_partitions=1)
    layout = jft.ShardingLayout(mesh, sample_axis=None)
    likelihood = jft.Gaussian(jnp.zeros(1), noise_cov_inv=lambda x: x)
    with pytest.raises(ValueError, match="mutually exclusive"):
        jft.OptimizeVI(
            likelihood,
            1,
            devices=jax.devices(),
            sharding=layout,
        )


@pytest.mark.skipif(jax.device_count() < 4, reason="requires four devices")
def test_linear_gaussian_inference_iteration_on_sample_space_mesh():
    n = 16
    points = jnp.linspace(0.0, 1.0, n)[:, None]
    graph = gp.build_graph(points, n0=4, k=2)
    covariance = gp.extras.matern_kernel(
        p=0,
        variance=1.0,
        cutoff=0.3,
        r_min=1e-4,
        r_max=2.0,
        n_bins=32,
        jitter=1e-5,
    )
    plan = gp.distributed.partition_graph(
        graph, np.repeat(np.arange(2), n // 2)
    )
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("sample", "space"))
    field = jft.GraphGPField(plan, covariance, mesh=mesh, prefix="f_")
    signal = jft.Model(
        lambda x: jnp.tanh(field(x)), domain=field.domain, init=field.init
    )
    likelihood = jft.Gaussian(
        jnp.zeros(field.target.shape), noise_cov_inv=lambda x: 4.0 * x
    ).amend(signal)
    layout = field.sharding_layout(sample_axis="sample")
    position = jft.Vector(likelihood.init(jax.random.key(5)))
    position = jax.device_put(position, layout.position_shardings(position))

    samples, state = jft.optimize_kl(
        likelihood,
        position,
        key=jax.random.key(6),
        n_total_iterations=1,
        n_samples=1,
        sample_mode="linear_resample",
        residual_map=jax.vmap,
        sharding=layout,
        draw_linear_kwargs=dict(
            cg=jft.conjugate_gradient.static_cg,
            cg_kwargs=dict(maxiter=5, absdelta=1e-4),
        ),
        kl_kwargs=dict(minimize_kwargs=dict(maxiter=1)),
    )

    assert state.nit == 1
    assert len(samples) == 2
    assert samples.pos.tree[field.excitation_name].sharding.spec == P("space")
    assert samples.samples.tree[field.excitation_name].sharding.spec == P(
        "sample", "space"
    )
