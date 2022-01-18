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
# Copyright(C) 2013-2021 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

############################################################
# Non-linear tomography
#
# The signal is a sigmoid-normal distributed field.
# The data is the field integrated along lines of sight that are
# randomly (set mode=0) or radially (mode=1) distributed
#
# Demo takes a while to compute
#############################################################

import sys

import numpy as np

import nifty8 as ift

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    master = comm.Get_rank() == 0
except ImportError:
    comm = None
    master = True


class SingleDomain(ift.LinearOperator):
    def __init__(self, domain, target):
        self._domain = ift.makeDomain(domain)
        self._target = ift.makeDomain(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        return ift.makeField(self._tgt(mode), x.val)


class IRGSpace(ift.StructuredDomain):
    """Represents non-equidistantly binned and non-periodic one-dimensional spaces.

    Parameters
    ----------
    coordinates : np.ndarray
        Must be sorted and strictly ascending.
    """

    _needed_for_hash = ["_coordinates"]

    def __init__(self, coordinates):
        bb = np.array(coordinates)
        if bb.ndim != 1:
            raise TypeError
        if np.any(np.diff(bb) <= 0.0):
            raise ValueError("Coordinates must be sorted and strictly ascending")
        self._coordinates = tuple(bb)

    def __repr__(self):
        return f"IRGSpace(shape={self.shape}, coordinates={self._coordinates})"

    @property
    def harmonic(self):
        """bool : Always False for this class."""
        return False

    @property
    def shape(self):
        return (len(self._coordinates),)

    @property
    def size(self):
        return self.shape[0]

    @property
    def scalar_dvol(self):
        return None

    @property
    def dvol(self):
        """Assume that the coordinates are the center of symmetric pixels."""
        return np.diff(self.binbounds())

    def binbounds(self):
        if len(self._coordinates) == 1:
            return np.array([-np.inf, np.inf])
        c = np.array(self._coordinates)
        bounds = np.empty(self.size + 1)
        bounds[1:-1] = c[:-1] + 0.5*np.diff(c)
        bounds[0] = c[0] - 0.5*(c[1] - c[0])
        bounds[-1] = c[-1] + 0.5*(c[-1] - c[-2])
        return bounds

    @property
    def distances(self):
        return np.diff(self._coordinates)

    @property
    def coordinates(self):
        return self._coordinates


class WienerIntegrations(ift.LinearOperator):
    """Operator that performs the integrations necessary for an integrated
    Wiener process.

    Parameters
    ----------
    time_domain : IRGSpace
        Domain that contains the temporal information of the process.

    remaining_domain : DomainTuple or Domain
        All integrations are handled independently for this domain.
    """
    def __init__(self, time_domain, remaining_domain):
        self._target = ift.makeDomain((time_domain, remaining_domain))
        dom = ift.UnstructuredDomain((2, time_domain.size - 1)), remaining_domain
        self._domain = ift.makeDomain(dom)
        self._volumes = time_domain.distances
        for _ in range(len(remaining_domain.shape)):
            self._volumes = self._volumes[..., np.newaxis]
        self._capability = self.TIMES | self.ADJOINT_TIMES


    def apply(self, x, mode):
        self._check_input(x, mode)
        first, second = (0,), (1,)
        from_second = (slice(1, None),)
        no_border = (slice(0, -1),)
        reverse = (slice(None, None, -1),)
        if mode == self.TIMES:
            x = x.val
            res = np.zeros(self._target.shape)
            res[from_second] = np.cumsum(x[second], axis=0)
            res[from_second] = (res[from_second] + res[no_border]) / 2 * self._volumes + x[first]
            res[from_second] = np.cumsum(res[from_second], axis=0)
        else:
            x = x.val_rw()
            res = np.zeros(self._domain.shape)
            x[from_second] = np.cumsum(x[from_second][reverse], axis=0)[reverse]
            res[first] += x[from_second]
            x[from_second] *= self._volumes / 2.0
            x[no_border] += x[from_second]
            res[second] += np.cumsum(x[from_second][reverse], axis=0)[reverse]
        return ift.makeField(self._tgt(mode), res)


def IntWProcessInitialConditions(a0, b0, wpop, irg_space=None):
    if ift.is_operator(wpop):
        tgt = wpop.target
    else:
        tgt = irg_space, a0.target[0]

    sdom = tgt[0]

    bc = _FancyBroadcast(tgt)
    factors = ift.full(sdom, 0)
    factors = np.empty(sdom.shape)
    factors[0] = 0
    factors[1:] = np.cumsum(sdom.distances)
    factors = ift.makeField(sdom, factors)
    res = bc @ a0 + ift.DiagonalOperator(factors, tgt, 0) @ bc @ b0

    if wpop is None:
        return res
    else:
        return wpop + res


class _FancyBroadcast(ift.LinearOperator):
    def __init__(self, target):
        self._target = ift.DomainTuple.make(target)
        self._domain = ift.DomainTuple.make(target[1])
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            res = np.broadcast_to(x.val[None], self._target.shape)
        else:
            res = np.sum(x.val, axis=0)
        return ift.makeField(self._tgt(mode), res)


class DomainChangerAndReshaper(ift.LinearOperator):
    def __init__(self, domain, target):
        self._domain = ift.DomainTuple.make(domain)
        self._target = ift.DomainTuple.make(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        tgt = self._tgt(mode)
        return ift.makeField(tgt, x.reshape(tgt.shape))


def random_los(n_los):
    starts = list(ift.random.current_rng().random((n_los, 2)).T)
    ends = list(ift.random.current_rng().random((n_los, 2)).T)
    return starts, ends


def radial_los(n_los):
    starts = list(ift.random.current_rng().random((n_los, 2)).T)
    ends = list(0.5 + 0*ift.random.current_rng().random((n_los, 2)).T)
    return starts, ends


def main():
    # Choose between random line-of-sight response (mode=0) and radial lines
    # of sight (mode=1)
    if len(sys.argv) == 2:
        mode = int(sys.argv[1])
    else:
        mode = 0

    # Preparing the filename string for store results
    filename = "getting_started_mf_mode_{}_".format(mode) + "{}.png"

    # Set up signal domain
    npix1, npix2 = 128, 128
    position_space = ift.RGSpace([npix1, npix2])
    sp1 = ift.RGSpace(npix1)
    sp2 = IRGSpace(np.arange(npix2)*sp1.scalar_dvol)

    # Set up signal model
    cfmaker = ift.CorrelatedFieldMaker('')
    cfmaker.add_fluctuations(sp1, (0.1, 1e-2), (2, .2), (.01, .5), (-4, 2.),
                             'amp1')
    cfmaker.set_amplitude_total_offset(0., (1e-2, 1e-6))
    i_0 = cfmaker.finalize()

    alpha = ift.SimpleCorrelatedField(sp1, 0., (1e-2, 1e-6), (0.1, 1e-2), (2, .2), (.01, .5), (-3, 1), prefix=f'alpha')

    # flex and asp parameters for the integrated Wiener Process
    flexibility, asperity = (1, 1), None  # (1e14, 1e14)

    n_freq_xi_fields = 2 * (sp2.size - 1)
    cfm = ift.CorrelatedFieldMaker("freq_xi", total_N=n_freq_xi_fields)
    cfm.set_amplitude_total_offset(0.0, None)
    cfm.add_fluctuations(sp1, (1, 1e-6), (1.2, 0.4), (0.2, 0.2), (-2, 0.5), dofdex=n_freq_xi_fields * [0])
    freq_xi = cfm.finalize(0)

    # Integrate over excitation fields
    intop = WienerIntegrations(sp2, sp1)
    freq_xi = DomainChangerAndReshaper(freq_xi.target, intop.domain) @ freq_xi
    broadcast = ift.ContractionOperator(intop.domain[0], None).adjoint
    broadcast_full = ift.ContractionOperator(intop.domain, 1).adjoint
    vol = sp2.distances

    flex = ift.LognormalTransform(*flexibility, "iwp_flexibility", 0)

    dom = intop.domain[0]
    vflex = np.empty(dom.shape)
    vflex[0] = vflex[1] = np.sqrt(vol)
    sig_flex = ift.makeOp(ift.makeField(dom, vflex)) @ broadcast @ flex
    sig_flex = broadcast_full @ sig_flex
    shift = np.ones(dom.shape)
    shift[0] = vol * vol / 12.0
    if asperity is None:
        shift = ift.DiagonalOperator(ift.makeField(dom, shift).sqrt(), intop.domain, 0)
        increments = shift @ (freq_xi * sig_flex)
    else:
        asp = ift.LognormalTransform(*asperity, "iwp_asperity", 0)
        vasp = np.empty(dom.shape)
        vasp[0] = 1
        vasp[1] = 0
        vasp = ift.DiagonalOperator(ift.makeField(dom, vasp), domain=broadcast.target, spaces=0)
        sig_asp = broadcast_full @ vasp @ broadcast @ asp
        shift = ift.makeField(intop.domain, np.broadcast_to(shift[..., None, None], intop.domain.shape))
        increments = freq_xi * sig_flex * (ift.Adder(shift) @ sig_asp).ptw("sqrt")

    logsky = IntWProcessInitialConditions(i_0, alpha, intop @ increments)

    normalized_amp = cfmaker.get_normalized_amplitudes()
    pspec1 = normalized_amp[0]**2
    # pspec2 = normalized_amp[1]**2
    DC = SingleDomain(logsky.target, position_space)

    # Apply a nonlinearity
    signal = DC @ ift.sigmoid(logsky)

    # Build the line-of-sight response and define signal response
    LOS_starts, LOS_ends = random_los(100) if mode == 0 else radial_los(100)
    R = ift.LOSResponse(position_space, starts=LOS_starts, ends=LOS_ends)
    signal_response = R(signal)

    # Specify noise
    data_space = R.target
    noise = .001
    N = ift.ScalingOperator(data_space, noise, float)

    # Generate mock signal and data
    mock_position = ift.from_random(signal_response.domain, 'normal')
    data = signal_response(mock_position) + N.draw_sample()

    plot = ift.Plot()
    plot.add(signal(mock_position), title='Ground Truth')
    plot.add(R.adjoint_times(data), title='Data')
    plot.add([pspec1.force(mock_position)], title='Power Spectrum 1')
    # plot.add([pspec2.force(mock_position)], title='Power Spectrum 2')
    plot.output(ny=2, nx=2, xsize=10, ysize=10, name=filename.format("setup"))

    # Minimization parameters
    ic_sampling = ift.AbsDeltaEnergyController(name='Sampling', deltaE=0.01, iteration_limit=100)
    ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=0.01, iteration_limit=35)
    minimizer = ift.NewtonCG(ic_newton, enable_logging=True)

    # Set up likelihood energy and information Hamiltonian
    likelihood_energy = ift.GaussianEnergy(mean=data, inverse_covariance=N.inverse) @ signal_response

    def callback(samples):
        s = ift.extra.minisanity(data, lambda x: N.inverse, signal_response, samples)
        if master:
            ift.logger.info(s)

    n_samples = 20
    n_iterations = 5
    samples = ift.optimize_kl(likelihood_energy, n_iterations, n_samples,
                              minimizer, ic_sampling, None, overwrite=True, comm=comm,
                              output_directory="getting_started_5_results",
                              ground_truth_position=mock_position,
                              plottable_operators={"signal": signal, "power spectrum 1": pspec1},
                                                   #"power spectrum 2": pspec2},
                              inspect_callback=callback)

    # Plotting
    filename_res = filename.format("results")
    plot = ift.Plot()
    mean, var = samples.sample_stat(signal)
    plot.add(mean, title="Posterior Mean")
    plot.add(ift.sqrt(var), title="Posterior Standard Deviation")

    n_samples = samples.n_samples
    plot.add(list(samples.iterator(pspec1)) + [samples.average(pspec1.log()).exp(),
              pspec1.force(mock_position)],
             title="Sampled Posterior Power Spectrum 1",
             linewidth=[1.]*n_samples + [3., 3.])
    # plot.add(list(samples.iterator(pspec2)) + [samples.average(pspec2.log()).exp(),
    #           pspec2.force(mock_position)],
    #          title="Sampled Posterior Power Spectrum 2",
    #          linewidth=[1.]*n_samples + [3., 3.])
    if master:
        plot.output(ny=2, nx=2, xsize=15, ysize=15, name=filename_res)
    print("Saved results as '{}'.".format(filename_res))


if __name__ == '__main__':
    main()
