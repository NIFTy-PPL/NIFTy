# %%

import os

nthreads = 1

os.environ[
    "XLA_FLAGS"
] = f"--xla_cpu_multi_thread_eigen={nthreads != 1} intra_op_parallelism_threads={nthreads}"
os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
os.environ["MKL_NUM_THREADS"] = str(nthreads)
os.environ["OMP_NUM_THREADS"] = str(nthreads)

import itertools
from functools import partial

import jax
import nifty8 as ift
import nifty8.re as jft
import numpy as np
import plotly.graph_objects as go
from jax import numpy as jnp
from jax import random
from matplotlib import pyplot as plt
from upy.detective import timeit
from upy.pprint import progress_bar as tqdm

jax.config.update("jax_enable_x64", True)

ift.set_nthreads(nthreads)


# %%
class Forward(jft.Model):
    def __init__(self, correlated_field):
        self._cf = correlated_field
        # Track a method with which a random input for the model. This is not
        # strictly required but is usually handy when building deep models.
        super().__init__(init=correlated_field.init)

    def __call__(self, x):
        # NOTE, any kind of masking of the output, non-linear and linear
        # transformation could be carried out here. Models can also combined and
        # nested in any way and form.
        return jnp.exp(self._cf(x))


# %%
def get_model_jft(dims, *, data_key=random.PRNGKey(42)):
    cfm = jft.CorrelatedFieldMaker("cf")
    cfm.set_amplitude_total_offset(offset_mean=2, offset_std=(1e-1, 3e-2))
    cfm.add_fluctuations(  # Axis over which the kernle is defined
        dims,
        distances=tuple(1.0 / d for d in dims),
        fluctuations=(1.0, 5e-1),
        loglogavgslope=(-3.0, 2e-1),
        flexibility=(1e0, 2e-1),
        asperity=(5e-1, 5e-2),
        prefix="ax1",
        non_parametric_kind="power",
    )
    correlated_field = cfm.finalize()  # forward model for a GP prior
    forward = Forward(correlated_field)

    k_f, k_n = random.split(data_key, 2)
    truth = forward(forward.init(k_f))
    data = random.poisson(k_n, truth)

    lh_jft = jft.Poissonian(data).amend(forward)
    return lh_jft


def get_model_nft(dims, *, data):
    dom = ift.RGSpace(dims, distances=tuple(1.0 / d for d in dims))
    cfm = ift.CorrelatedFieldMaker("cf")
    cfm.set_amplitude_total_offset(offset_mean=2, offset_std=(1e-1, 3e-2))
    cfm.add_fluctuations(  # Axis over which the kernle is defined
        dom,
        fluctuations=(1.0, 5e-1),
        loglogavgslope=(-3.0, 2e-1),
        flexibility=(1e0, 2e-1),
        asperity=(5e-1, 5e-2),
        prefix="ax1",
    )
    correlated_field_ift = cfm.finalize(prior_info=0)
    forward_ift = correlated_field_ift.exp()

    lh_nft = (
        ift.PoissonianEnergy(ift.Field(ift.DomainTuple.make(dom), np.array(data)))
        @ forward_ift
    )
    return lh_nft


def generic_lh_metric(lh, p, t):
    return lh.metric(p, t)


# %%
# dim_range = (16, 10**3, 10)
dim_range = (16, 10**4, 20)
all_dims = [(i,) * 2 for i in np.geomspace(*dim_range).astype(int)]

all_devices = list(set(jax.devices()) | set(jax.devices(backend="cpu")))
all_t_jft = {dev.device_kind: [] for dev in all_devices}
all_t_nft = {dev.device_kind: [] for dev in all_devices}
key = random.PRNGKey(123)

for dev, dims in tqdm(
    itertools.product(all_devices, all_dims), total=len(all_devices) * len(all_dims)
):
    lh_jft = get_model_jft(dims)
    pos = jft.Vector(lh_jft.init(key))
    pos = jax.device_put(pos, device=dev)
    lh_met = partial(jax.jit(generic_lh_metric, device=dev), lh_jft)

    # Warm-up
    lh_met_wm = lh_met(pos, pos).tree
    all_t_jft[dev.device_kind].append(
        timeit(lambda: jax.block_until_ready(lh_met(pos, pos)))
    )

    if dev.device_kind.lower() == "cpu":
        lh_nft = get_model_nft(dims, data=lh_jft.likelihood.data)
        pos_nft = ift.MultiField.from_dict(
            {
                k: ift.Field(d, np.array(v))
                for (k, d), v in zip(lh_nft.domain.items(), pos.tree.values())
            }
        )
        lh_metric_nft = lh_nft(
            ift.Linearization.make_var(pos_nft, want_metric=True)
        ).metric
        all_t_nft[dev.device_kind].append(timeit(lambda: lh_metric_nft(pos_nft)))
        if np.prod(dims) < 1e6:
            jax.tree_map(
                partial(np.testing.assert_allclose, atol=1e-6, rtol=1e-7),
                lh_met_wm,
                lh_metric_nft(pos_nft).val,
            )

# %%
devs_nm = "+".join(dev.device_kind for dev in all_devices)
np.save(
    f"benchmark_nthreads={nthreads}_devices={devs_nm}.npy",
    dict(all_dims=all_dims, all_t_jft=all_t_jft, all_t_nft=all_t_nft),
    allow_pickle=True,
)

# %%
# The below is a fancy version of
# ```
# fig, ax = plt.subplots(1, 1)
# ax.plot([np.prod(d) for d in all_dims], [t.time for t in all_t_jft], label="NIFTy.re")
# ax.plot([np.prod(d) for d in all_dims], [t.time for t in all_t_nft], label="NIFTy")
# ax.set_xlabel()
# ax.set_ylabel()
# ax.legend()
# ax.set_xscale("log")
# ax.set_yscale("log")
# plt.show()
# ```

fig = go.Figure()
for nm in all_t_jft.keys():
    fig.add_trace(
        go.Scatter(
            x=np.array([np.prod(d) for d in all_dims]),
            y=np.array([t.time for t in all_t_jft[nm]]),
            mode="lines+markers",
            name=f"{nm.upper()} NIFTy.re",
        )
    )
assert len(all_t_nft)
nm = list(all_t_nft.keys())[0]
fig.add_trace(
    go.Scatter(
        x=np.array([np.prod(d) for d in all_dims]),
        y=np.array([t.time for t in all_t_nft[nm]]),
        mode="lines+markers",
        name=f"NIFTy",
    )
)

fig.update_layout(
    showlegend=True,
    template="plotly_white",
    xaxis_title="number of dimensions",
    yaxis_title="time [s]",
    width=720,
    height=480,
)
fig.update_xaxes(type="log")
fig.update_yaxes(type="log")
devs_nm = "+".join(dev.device_kind for dev in all_devices)
fig.show()
fig.write_html(f"benchmark_nthreads={nthreads}_devices={devs_nm}.html")
fig.write_image(f"benchmark_nthreads={nthreads}_devices={devs_nm}.png")
