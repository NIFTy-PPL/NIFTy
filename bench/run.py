# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2025 Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import csv
import subprocess

import matplotlib.pyplot as plt
import numpy as np

npixs = 2**np.linspace(6, 10, num=5)
npixs = np.round(npixs).astype(int)

for npix in npixs:
    for method in ["numpy", "numpy-8threads", "cupy"]:
        subprocess.run(["python3", "gaussian_energy.py", str(npix), method])

with open('benchmark_data.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',',)
    header = next(reader)
    data = []
    for row in reader:
        data.append(row)


def get_col(d, ind, dtype=lambda x: x):
    return [dtype(x[ind]) for x in d]

methods = set(get_col(data, 0))
xs = np.sort(np.unique(np.array(get_col(data, 1, int))))
measurements = header

for imeas, measurement in enumerate(measurements):
    for method in methods:
        ind = np.array([x == method for x in get_col(data, 0)])
        xs = np.array(get_col(data, 1, int))[ind]
        ys = np.array(get_col(data, 2+imeas, float))[ind]

        sort_ind = np.argsort(xs)
        xs = xs[sort_ind]
        ys = ys[sort_ind]

        plt.plot(xs, ys, "o-", label=method, alpha=0.6)
    plt.legend()
    plt.title(measurement)
    plt.ylim([1e-4, 2])
    #plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("npix")
    plt.ylabel("Wall time [s]")
    plt.tight_layout()
    plt.savefig(f"{measurement}.png")
    plt.close()
