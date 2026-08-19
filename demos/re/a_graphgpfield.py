"""Small end-to-end spatially sharded GraphGP reconstruction.

For a CPU-only smoke test, start Python with
``XLA_FLAGS=--xla_force_host_platform_device_count=4``.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

import graphgp as gp
import nifty.re as jft


jax.config.update("jax_enable_x64", True)

if jax.device_count() < 4:
    raise RuntimeError("this demo requires at least four JAX devices")

# A tiny two-dimensional graph; the same interface applies to much larger fields.
shape = (4, 4)
coordinates = jnp.stack(
    jnp.meshgrid(
        jnp.linspace(0.0, 1.0, shape[0]),
        jnp.linspace(0.0, 1.0, shape[1]),
        indexing="ij",
    ),
    axis=-1,
).reshape((-1, 2))
graph = gp.build_graph(coordinates, n0=4, k=2)
owners = np.repeat(np.arange(4), len(coordinates) // 4)
plan = gp.distributed.partition_graph(graph, owners, output_shape=shape)

# The singleton sample axis keeps all four devices available for spatial shards.
mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(1, 4), ("sample", "space"))
covariance = jft.MaternCovarianceModel(
    Ndim=2,
    r_min=1e-2,
    r_max=2.0,
    variance=(1.0, 0.1),
    lengthscale=(0.3, 0.05),
    negloglogslope=3.0,
    Ninterp=32,
    mode="graphgp",
    prefix="dust_cov_",
    enforce_monotonicity=False,
)
field = jft.GraphGPField(
    plan, covariance, mesh=mesh, prefix="dust_", offset=0.0
)
signal = jft.Model(
    lambda x: jnp.tanh(field(x)), domain=field.domain, init=field.init
)
likelihood = jft.Gaussian(
    data=jnp.zeros(shape), noise_cov_inv=lambda x: 4.0 * x
).amend(signal)

_, key_init, key_tangent, key_inference = jax.random.split(jax.random.key(7), 4)
position = jft.Vector(likelihood.init(key_init))
layout = field.sharding_layout(sample_axis="sample")
position = jax.device_put(position, layout.position_shardings(position))
tangent = jft.random_like(key_tangent, position)

print("likelihood:", likelihood(position))
print("metric norm:", jft.norm(likelihood.metric(position, tangent)))

samples, state = jft.optimize_kl(
    likelihood,
    position,
    key=key_inference,
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

result = field(samples.pos)
assert result.sharding.spec == P("space")
print("iteration:", state.nit)
print("field sharding:", result.sharding)
