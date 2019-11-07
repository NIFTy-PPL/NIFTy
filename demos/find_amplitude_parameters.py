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
# Copyright(C) 2013-2019 Max-Planck-Society
# Author: Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

import nifty5 as ift
import matplotlib.pyplot as plt


def _default_pspace(dom):
    return ift.PowerSpace(dom.get_default_codomain())


if __name__ == '__main__':
    np.random.seed(42)
    fa = ift.CorrelatedFieldMaker()
    n_samps = 20
    slope_means = [-2, -3]
    fa.add_fluctuations(_default_pspace(ift.RGSpace(128, 0.1)), 10, 2, 1, 1e-6,
                        2, 1e-6, slope_means[0], 0.2, 'spatial')
    # fa.add_fluctuations(_default_pspace(ift.RGSpace((128, 64))), 10, 2, 1,
    #                     1e-6, 2, 1e-6, slope_means[0], 0.2, 'spatial')
    fa.add_fluctuations(_default_pspace(ift.RGSpace(32)), 10, 5, 1, 1e-6, 2,
                        1e-6, slope_means[1], 1, 'freq')
    correlated_field = fa.finalize(10, 0.1, '')
    amplitudes = fa.amplitudes
    plt.style.use('seaborn-notebook')

    tgt = correlated_field.target
    if len(tgt.shape) == 1:
        fig, axes = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(20, 10)
    else:
        fig, axes = plt.subplots(nrows=3, ncols=3)
        fig.set_size_inches(20, 16)
    axs = (ax for ax in axes.ravel())
    for ii, aa in enumerate(amplitudes):
        ax = next(axs)
        pspec = aa**2
        ax.set_xscale('log')
        ax.set_yscale('log')
        for _ in range(n_samps):
            fld = pspec(ift.from_random('normal', pspec.domain))
            klengths = fld.domain[0].k_lengths
            ycoord = fld.to_global_data_rw()
            ycoord[0] = ycoord[1]
            ax.plot(klengths, ycoord, alpha=1)

        ymin, ymax = ax.get_ylim()
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
        lbl = 'Mean slope (k^{})'.format(2*slope_means[ii])
        for fac in np.linspace(np.log(ymin), np.log(ymax**2/ymin)):
            xs = np.linspace(np.amin(klengths[1:]), np.amax(klengths[1:]))
            ys = xs**(2*slope_means[ii])*np.exp(fac)
            xs = np.insert(xs, 0, 0)
            ys = np.insert(ys, 0, ys[0])
            ax.plot(xs, ys, zorder=1, color=color, linewidth=0.3, label=lbl)
            lbl = None

        ax.set_ylim([ymin, ymax])
        ax.set_xlim([None, np.amax(klengths)])
        ax.legend()

    if len(tgt.shape) == 2:
        foo = []
        for ax in axs:
            pos = ift.from_random('normal', correlated_field.domain)
            fld = correlated_field(pos).to_global_data()
            foo.append((ax, fld))
        mi, ma = np.inf, -np.inf
        for _, fld in foo:
            mi = min([mi, np.amin(fld)])
            ma = max([ma, np.amax(fld)])
        nxdx, nydy = tgt.shape
        if len(tgt) == 2:
            nxdx *= tgt[0].distances[0]
            nydy *= tgt[1].distances[0]
        else:
            nxdx *= tgt[0].distances[0]
            nydy *= tgt[0].distances[1]
        for ax, fld in foo:
            im = ax.imshow(fld.T,
                           extent=[0, nxdx, 0, nydy],
                           aspect='auto',
                           origin='lower',
                           vmin=mi,
                           vmax=ma)
        fig.colorbar(im, ax=axes.ravel().tolist())
    elif len(tgt.shape) == 1:
        ax = next(axs)
        flds = []
        for _ in range(n_samps):
            pos = ift.from_random('normal', correlated_field.domain)
            ax.plot(correlated_field(pos).to_global_data())

    plt.savefig('correlated_fields.png')
    plt.close()
