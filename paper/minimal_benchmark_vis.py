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
pretty_name_map = {
    "all_t_jft": "NIFTy.re",
    "all_t_nft": "NIFTy",
}
magic_gpu_key = "GPU NIFTy.re"

savestate = {}
for bench, fn in benchmark_files.items():
    timings = np.load(fn, allow_pickle=True).item()
    all_dims = timings.pop("all_dims")
    for backend, timings_by_platform in timings.items():
        assert isinstance(timings_by_platform, dict)
        for platform, timings in timings_by_platform.items():
            nm = f"{platform.upper()} w/ {bench} core(s) {pretty_name_map[backend]}"
            if platform.lower().startswith("nvidia"):
                if backend != "all_t_jft":
                    if len(timings) != 0:
                        print(f"Skipping invalid {bench=} {backend=} {platform=}")
                    continue
                nm = magic_gpu_key
                if nm in savestate:
                    print(f"Skipping duplicate key {nm!r}")
                    continue
            print(f"Adding key {nm!r:36s} w/ {bench=} {backend=} {platform=}")
            savestate[nm] = (all_dims, timings)
savestate = {k: savestate[k] for k in sorted(savestate.keys())}

# %%
fig = go.Figure()
n_gpu_plots = 0
for pretty_name, (all_dims, timings) in savestate.items():
    fig.add_trace(
        go.Scatter(
            x=np.array([np.prod(d) for d in all_dims]),
            y=np.array([t.time for t in timings]),
            mode="lines+markers",
            name=pretty_name,
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
    margin=dict(t=5, b=5, l=5, r=5),
)
fig.update_xaxes(type="log")
fig.update_yaxes(type="log")
fn_out_stem = "benchmark_nthreads=1+8_devices=cpu+gpu"
fig.write_html(fn_out_stem + ".html")
fig.write_image(fn_out_stem + ".png", scale=4)
fig.show()
