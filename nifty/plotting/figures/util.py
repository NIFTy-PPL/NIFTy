from nifty.plotting.plots.private import _Plot2D, _Plot3D
from nifty.plotting.plots import ScatterGeoMap


def validate_plots(data):
    if not data:
        raise Exception('Error: no plots given')

    if type(data) != list:
        data = [data]

    if isinstance(data[0], _Plot2D):
        kind = _Plot2D
    elif isinstance(data[0], _Plot3D):
        kind = _Plot3D
    elif isinstance(data[0], ScatterGeoMap):
        kind = ScatterGeoMap
    else:
        kind = None

    if kind:
        for plt in data:
            if not isinstance(plt, kind):
                raise Exception(
                    """Error: Plots are not of the right kind!
                    Compatible types are:
                     - Scatter2D and HeatMap
                     - Scatter3D
                     - ScatterMap""")
    else:
        raise Exception('Error: plot type unknown')

    return kind, data
