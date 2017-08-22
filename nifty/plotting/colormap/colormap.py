from __future__ import division
from builtins import str
from nifty.plotting.plotly_wrapper import PlotlyWrapper


class Colormap(PlotlyWrapper):
    def __init__(self, name, red, green, blue):
        self.name = name
        self.red = red
        self.green = green
        self.blue = blue

    def validate_map(self):
        def validade(m):
            #TODO: implement validation
            pass

    # no discontinuities only
    @staticmethod
    def from_matplotlib_colormap_internal(name, mpl_cmap):
        red = [(c[0], c[2]) for c in mpl_cmap['red']]
        green = [(c[0], c[2]) for c in mpl_cmap['green']]
        blue = [(c[0], c[2]) for c in mpl_cmap['blue']]
        return Colormap(name, red, green, blue)

    def to_plotly(self):
        r, g, b = 0, 0, 0
        converted = list()
        prev_split, prev_r, prev_g, prev_b = 0., 0., 0., 0.

        while prev_split < 1:
            next_split = min(self.red[r][0], self.blue[b][0], self.green[g][0])

            if next_split == self.red[r][0]:
                red_val = self.red[r][1]
                r += 1
            else:
                slope = (self.red[r][1]-prev_r) / (self.red[r][0] - prev_split)
                y = prev_r - slope * prev_split
                red_val = slope * next_split + y

            if next_split == self.green[g][0]:
                green_val = self.green[g][1]
                g += 1
            else:
                slope = ((self.green[g][1] - prev_g) /
                         (self.green[g][0] - prev_split))
                y = prev_g - slope * prev_split
                green_val = slope * next_split + y

            if next_split == self.blue[b][0]:
                blue_val = self.blue[b][1]
                b += 1
            else:
                slope = ((self.blue[b][1] - prev_b) /
                         (self.blue[b][0] - prev_split))
                y = prev_r - slope * prev_split
                blue_val = slope * next_split + y

            prev_split = next_split
            prev_r = red_val
            prev_g = green_val
            prev_b = blue_val

            converted.append([next_split,
                              'rgb(' +
                              str(int(red_val*255)) + "," +
                              str(int(green_val*255)) + "," +
                              str(int(blue_val*255)) + ")"])

        return converted





