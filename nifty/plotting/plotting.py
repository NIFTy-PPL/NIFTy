import os
from PIL import Image

import plotly.offline as ply_offline
import plotly.plotly as ply


def plot(figure, filename=None, interactive=False):
    if not filename:
        filename = os.path.abspath('/tmp/temp-plot.html')
    if interactive:
        try:
            __IPYTHON__
            ply_offline.init_notebook_mode(connected=True)
            ply_offline.iplot(figure._to_plotly(), filename=filename)
        except NameError:
            ply_offline.plot(figure._to_plotly(), filename=filename)
            raise Warning('IPython not active! Running without interactive mode.')
    else:
        ply_offline.plot(figure._to_plotly(), filename=filename)


def plot_image(figure, filename=None, show=False):
    if not filename:
        filename = os.path.abspath('temp-plot.jpeg')
    ply_obj = figure._to_plotly()
    ply.image.save_as(ply_obj, filename=filename)
    if show:
        img = Image.open(filename)
        img.show()



