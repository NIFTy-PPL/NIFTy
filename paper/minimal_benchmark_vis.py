# %%
import numpy as np
import plotly.graph_objects as go

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

# %%
fn_benchmark = "benchmark_nthreads=1_devices=cpu+NVIDIA A100-SXM4-80GB.npy"
savestate = np.load(fn_benchmark, allow_pickle=True).item()

fig = go.Figure()
for nm in savestate["all_t_jft"].keys():
    pretty_name = nm if not nm.lower().startswith("nvidia") else "GPU"
    fig.add_trace(
        go.Scatter(
            x=np.array([np.prod(d) for d in savestate["all_dims"]]),
            y=np.array([t.time for t in savestate["all_t_jft"][nm]]),
            mode="lines+markers",
            name=f"{pretty_name.upper()} NIFTy.re",
        )
    )
assert len(savestate["all_t_nft"])
nm = list(savestate["all_t_nft"].keys())[0]
fig.add_trace(
    go.Scatter(
        x=np.array([np.prod(d) for d in savestate["all_dims"]]),
        y=np.array([t.time for t in savestate["all_t_nft"][nm]]),
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
fig.write_html(fn_benchmark.replace(".npy", ".html"))
fig.write_image(fn_benchmark.replace(".npy", ".png"), scale=4)
