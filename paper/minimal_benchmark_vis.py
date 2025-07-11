# %%
import numpy as np
import plotly.express as px
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
    "1": "benchmark_nthreads=1_devices=NVIDIA A100-SXM4-80GB+cpu.npy",
    "8": "benchmark_nthreads=8_devices=NVIDIA A100-SXM4-80GB+cpu.npy",
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
            time_m = list(map(lambda x: x["median"], timings))
            time_q16 = list(map(lambda x: x["q16"], timings))
            time_q84 = list(map(lambda x: x["q84"], timings))
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
            savestate[nm] = (all_dims, time_m, time_q16, time_q84)
savestate = {k: savestate[k] for k in sorted(savestate.keys())}

# %%
fig = go.Figure()
n_gpu_plots = 0
symbol_cycle = ("circle", "cross", "x", "star-triangle-up", "square", "y-down")
dash_cycle = ("solid", "dot", "dash", "longdash", "dashdot", "longdashdot")
color_map = {
    "NIFTy": px.colors.qualitative.G10[0],
    "NIFTy.re": px.colors.qualitative.G10[1],
}
for i, (pretty_name, (all_dims, tm, tq16, tq84)) in enumerate(savestate.items()):
    sym, dash = symbol_cycle[i % len(savestate)], dash_cycle[i % len(savestate)]
    color = color_map[pretty_name.split(" ")[-1]]
    fig.add_trace(
        go.Scatter(
            x=np.array([np.prod(d) for d in all_dims]),
            y=np.array(tm),
            error_y=dict(
                type="data",
                symmetric=False,
                array=np.array(tq84) - np.array(tm),
                arrayminus=np.array(tm) - np.array(tq16),
                width=0,
            ),
            mode="lines+markers",
            line=dict(dash=dash, color=color),
            marker=dict(symbol=sym, color=color),
            name=pretty_name,
            opacity=0.9,
        )
    )
fig.update_layout(
    legend=dict(x=0.05, y=1),
    showlegend=True,
    template="plotly_white",
    xaxis_title="number of dimensions",
    yaxis_title="time [s]",
    width=720,
    height=400,
    margin=dict(t=5, b=5, l=5, r=5),
)
fig.update_xaxes(type="log")
fig.update_yaxes(type="log")
fn_out_stem = "benchmark_nthreads=1+8_devices=cpu+gpu"
fig.write_html(fn_out_stem + ".html")
fig.write_image(fn_out_stem + ".png", scale=4)
fig.show()
