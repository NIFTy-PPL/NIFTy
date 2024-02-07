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
benchmark_files = {
    "1": "benchmark_nthreads=1_devices=cpu+NVIDIA A100-SXM4-80GB.npy",
    "8": "benchmark_nthreads=8_devices=NVIDIA A100-SXM4-80GB.npy",
}
savestate = {}
for prefix, fn in benchmark_files.items():
    savestate[prefix] = np.load(fn, allow_pickle=True).item()

# Make the GPU timings the first entry in the dictionary as to automatically
# order the graphs in the following figure nicely
for p, svt in savestate.items():
    for nm in svt["all_t_jft"].keys():
        gpu_key = None
        if nm.lower().startswith("nvidia"):
            gpu_key = nm
        gpu_key = set((gpu_key,)) if gpu_key is not None else set()
    new_key_order = set(svt["all_t_jft"].keys()) - gpu_key
    new_key_order = tuple(gpu_key) + tuple(new_key_order)
    svt["all_t_jft"] = {k: svt["all_t_jft"][k] for k in new_key_order}

# %%
fig = go.Figure()
n_gpu_plots = 0
for prefix, svt in savestate.items():
    for nm in svt["all_t_jft"].keys():
        pretty_name = nm.upper() + f" w/ {prefix} core(s)"
        if nm.lower().startswith("nvidia"):
            pretty_name = "GPU"
            n_gpu_plots += 1
            if n_gpu_plots > 1:
                continue
        fig.add_trace(
            go.Scatter(
                x=np.array([np.prod(d) for d in svt["all_dims"]]),
                y=np.array([t.time for t in svt["all_t_jft"][nm]]),
                mode="lines+markers",
                name=f"NIFTy.re on {pretty_name}",
            )
        )
    assert len(svt["all_t_nft"])
    nm = list(svt["all_t_nft"].keys())[0]
    fig.add_trace(
        go.Scatter(
            x=np.array([np.prod(d) for d in svt["all_dims"]]),
            y=np.array([t.time for t in svt["all_t_nft"][nm]]),
            mode="lines+markers",
            name=f"NIFTy w/ {prefix} core(s)",
        )
    )

fig.update_layout(
    legend=dict(x=0.05, y=1),
    showlegend=True,
    template="plotly_white",
    xaxis_title="number of dimensions",
    yaxis_title="time [s]",
    width=720,
    height=480,
)
fig.update_xaxes(type="log")
fig.update_yaxes(type="log")
fn_out_stem = "benchmark_nthreads=1+8_devices=cpu+gpu"
fig.write_html(fn_out_stem + ".html")
fig.write_image(fn_out_stem + ".png", scale=4)
