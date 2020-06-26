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
# Copyright(C) 2019-2020 Max-Planck-Society

import numpy as np

from ..domain_tuple import DomainTuple
from ..domains.rg_space import RGSpace
from ..domains.unstructured_domain import UnstructuredDomain
from ..field import Field
from ..linearization import Linearization
from ..logger import logger
from ..multi_field import MultiField
from ..operators.harmonic_operators import FFTOperator
from ..operators.linear_operator import LinearOperator
from ..operators.scaling_operator import ScalingOperator
from ..probing import StatCalculator
from ..random import Context, current_rng, spawn_sseq
from ..sugar import makeField, makeOp
from ..utilities import allreduce_sum, get_MPI_params_from_comm, shareRange


def _mean(fld, dom):
    result = {}
    for key in fld.keys():
        mean = fld[key].val.mean(axis=-1)
        result[key[:-2]] = makeField(dom[key[:-2]], mean)
    return MultiField.from_dict(result, dom)


def _var(fld, dom):
    result = {}
    for key in fld.keys():
        var = fld[key].val.var(axis=-1)
        result[key[:-2]] = makeField(dom[key[:-2]], var)
    return MultiField.from_dict(result, dom)


def _standardized_sample_field(samples):
    di = {}
    dom = samples[0].domain
    fld = _sample_field(samples)
    mean = _mean(fld, dom)
    var = _var(fld, dom)
    for key in dom.keys():
        sub_fld = fld[key + '_t']
        sub_dom = sub_fld.domain
        sub_fld = (sub_fld.val - mean[key].val[..., np.newaxis])/(
            var[key].val**0.5)[..., np.newaxis]
        di[key + '_t'] = makeField(sub_dom, sub_fld)
    return di


def _sample_field(samples):
    di = {}
    time_domain = RGSpace(len(samples))
    dom = samples[0].domain
    for key in dom.keys():
        fld = np.empty(dom[key].shape + time_domain.shape)
        fld_dom = DomainTuple.make(dom[key]._dom + (time_domain,))
        for i in range(time_domain.shape[0]):
            fld[..., i] = (samples[i][key]).val
        di[key + '_t'] = makeField(fld_dom, fld)
    return di


class HMC_chain:
    """Class for individual chains to perform the Hamiltonian Monte Carlo sampling.

    Parameters
    -----------
    V: EnergyOperator
        The problem Hamiltonian, used as potential energy in the Hamiltonian
        Dynamics of HMC.
    position:  Fields/MultiFields
        The position the chains are initialized.
    M: DiagonalOperator
        The mass matrix for the momentum term in the Hamiltonian dynamics.
        If not set, a unit matrix is assumed. Default: None
    steplength: Float
        The length of the steps in the leapfrog integration. This should be
        tuned to achieve reasonable acceptance for the given problem.
        Default: 0.003
    steps: positive Integer
        The number of leapfrog integration steps for the next sample.
        Default: 10
    """
    def __init__(self, V, position, M=None, steplength=0.003, steps=10, sseq=None):
        if sseq is None:
            raise RuntimeError
        if M is None:
            M = ScalingOperator(position.domain, 1)
        self._position = position
        self.samples = []
        self._M = M
        self._V = V
        self._steplength = steplength
        self._steps = steps
        self._energies = []
        self._accepted = []
        self._current_acceptance = []
        self._sseq = sseq

    def sample(self, N):
        """ The method to draw a set of samples.

        Parameters
        -----------
        N: positive Integer
        The number of samples to be drawn.
        """
        for i in range(N):
            self._sample()
            logger.info(f'iteration: {i} acceptance: {self._current_acceptance[-1]} steplength: {self._steplength}')

    def warmup(self, N, preferred_acceptance=0.6, keep=False):
        """ Performing a warmup by tuning the steplength
         to achieve a certain acceptance rate and estimating the mass matrix.

        Parameters
        -----------
        N: positive Integer
            The number of warmup samples to be drawn.
        preferred_acceptance: Float
            The acceptance rate according to which the stepsize is tuned.
            Default: 0.6
        keep: Boolean
            Whether to keep the drawn samples or discard them. Default: False
        """
        for i in range(N):
            self._sample()
            self._tune_parameters(preferred_acceptance)
            logger.info(f'WARMUP: {i} acceptance: {self._current_acceptance[-1]} steplength: {self._steplength}')
        sc = StatCalculator()
        for sample in self.samples:
            sc.add(sample)
        self.M = makeOp(sc.var).inverse
        if not keep:
            self.samples = []

    def estimate_quantity(self, function):
        """ Estimates the result of a function over all samples of the chains.

        Parameters
        -----------
        function: Function
            The function to be evaluated and averaged over the samples.

        Returns
        -----------
        mean, var : Tuple
            The mean and variance over the samples.
        """
        sc = StatCalculator()
        for sample in self.samples:
            sc.add(function(sample))
        return sc.mean, sc.var

    def _sample(self):
        """Draws one sample according to the HMC algorithm."""
        tmp = self._sseq.spawn(2)[1]
        with Context(tmp):
            momentum = self._M.draw_sample_with_dtype(dtype=np.float64)

        new_position, new_momentum = self._integrate(momentum)
        self._accepting(momentum, new_position, new_momentum)
        self._update_acceptance()

    def _integrate(self, momentum):
        """Performs the leapfrog integration of the equations of motion.

        Parameters
        -----------
        momentum: Field or Multifield
            The momentum vector in the Hamilton equations.
        """
        position = self._position
        for i in range(self._steps):
            position, momentum = self._leapfrog(position, momentum)
        return position, momentum

    def _leapfrog(self, position, momentum):
        """Performs one leapfrog integration step.

        Parameters
        -----------
        position: Field or Multifield
            The position vector in the Hamilton equations.
        momentum: Field or Multifield
            The momentum vector in the Hamilton equations.
        """
        lin = Linearization.make_var(position)
        gradient = self._V(lin).gradient
        momentum = momentum - self._steplength/2.*gradient
        position = position + self._steplength*self._M.inverse(momentum)
        lin = Linearization.make_var(position)
        gradient = self._V(lin).gradient
        momentum = momentum - self._steplength/2.*gradient
        return position, momentum

    def _accepting(self, momentum, new_position, new_momentum):
        """ Decides whether to accept or decline a new position according to
        Metropolis-Hastings.

        The current position is then stored as new sample.

        Parameters
        -----------
        momentum: Field or Multifield
            The old momentum vector in the Hamilton equations.
        new_position: Field or Multifield
            The new position vector after evolving the equations of motion.
        new_momentum: Field or Multifield
            The new momentum vector after evolving the equations of motion.
        """
        energy = self._V(self._position).val + (
            0.5*momentum.vdot(self._M.inverse(momentum))).val
        new_energy = self._V(new_position).val + (
            0.5*new_momentum.vdot(self._M.inverse(new_momentum))).val
        if new_energy < energy:
            self._position = new_position
            accept = 1
        else:
            rate = np.exp(energy - new_energy)
            if np.isnan(rate):
                return
            rng = current_rng()
            accept = rng.binomial(1, rate)
            if accept:
                self._position = new_position
        self._accepted.append(accept)
        self.samples.append(self._position)
        self._energies.append(energy)

    def _update_acceptance(self):
        """Calculates the current acceptance rate based on the last ten samples."""
        current_accepted = self._accepted[-10:]
        current_accepted = np.array(current_accepted)
        current_acceptance = np.mean(current_accepted)
        self._current_acceptance.append(current_acceptance)

    def _tune_parameters(self, preferred_acceptance):
        """Increases or decreases the steplength in the leapfrog integration
        based on the current acceptance rate to aim for the preferred rate.

        Parameters
        -----------
        preferred_acceptance: Float
            The preferred acceptance rate.
        """
        if self._current_acceptance[-1] < preferred_acceptance:
            self._steplength *= 0.99
        else:
            self._steplength *= 1.01

    @property
    def ESS(self):
        """The effective sample size over all samples of the chain.

        Returns
        -----------
        ESS: MultiField
            The effective sample size of all model parameters of the chain.
        """
        sample_field = _standardized_sample_field(self.samples)
        result = {}
        for key, sf in sample_field.items():
            AFC = ACF_Selector(sf.domain, len(self.samples))
            FFT = FFTOperator(sf.domain, space=len(sf.domain._dom) - 1)
            h = FFT(sf)
            autocorr = AFC(FFT.inverse(h.conjugate()*h)).real

            addaxis = False
            if len(autocorr.shape) == 1:  # FIXME ?
                autocorr = autocorr.val.reshape((1,) + autocorr.shape)
                addaxis = True
            else:
                autocorr = autocorr.val
            cum_field = np.cumsum(autocorr, axis=-1)
            correlation_length = np.argmax(autocorr < 0, axis=-1)
            indices = np.where(np.ones(cum_field[..., 0].shape))
            indices += (correlation_length.flatten() - 1,)
            integr_corr = cum_field[indices] - 1
            ESS = len(self.samples)/(1 + 2*integr_corr)
            if addaxis:
                result[key[:-2]] = Field(self.samples[0].domain[key[:-2]], ESS[0])
            else:
                result[key[:-2]] = Field(self.samples[0].domain[key[:-2]],
                                         ESS.reshape(correlation_length.shape))
        return MultiField.from_dict(result)

    def mean(self):
        """The mean over all samples of the chain.

        Returns
        -----------
        mean: Field or MultiField
            The mean over all samples of the chain.
        """
        return _mean(self._sample_field(), self._position.domain)


class HMC_Sampler:
    """The sampler class, managing chains and the computations of diagnostics.

    Parameters
    -----------
    V: EnergyOperator
        The problem Hamiltonian, used as potential energy in the Hamiltonian
        Dynamics of HMC.
    initial_position: List of Fields/MultiFields
        The position the chains are initialized.
    chains: positive Integer
        The number of chains. Default: 1
    M: DiagonalOperator
        The mass matrix for the momentum term in the Hamiltonian dynamics.
        If not set, a unit matrix is assumed. Default: None
    steplength: Float
        The length of the steps in the leapfrog integration. This should be
        tuned to achieve reasonable acceptance for the given problem.
        Default: 0.003
    steps: positive Integer
        The number of leapfrog integration steps for the next sample.
        Default: 10
    """
    def __init__(self, V, initial_position, chains=1, M=None, steplength=0.003, steps=10, comm=None):
        self._M = M
        self._V = V
        self._dom = initial_position[0].domain  # FIXME temporary!
        self._steplength = steplength
        self._steps = steps
        self._N_chains = chains
        sseq = spawn_sseq(self._N_chains)
        self._local_chains = []
        self._comm = comm
        ntask, rank, _ = get_MPI_params_from_comm(self._comm)
        lo, hi = shareRange(self._N_chains, ntask, rank)
        for i in range(lo, hi):
            self._local_chains.append(
                HMC_chain(self._V, initial_position[i], self._M,
                          self._steplength, self._steps, sseq[i]))

    def sample(self, N):
        """The method to draw a set of samples in every chain.

        Parameters
        -----------
        N: positive Integer
            The number of samples to be drawn in every chain.
        """
        for chain in self._local_chains:
            chain.sample(N)

    def warmup(self, N, preferred_acceptance=0.6, keep=False):
        """Performing a warmup by tuning the steplength to achieve a certain
        acceptance rate and estimating the mass matrix.

        Parameters
        -----------
        N: positive Integer
            The number of warmup samples to be drawn in every chain.
        preferred_acceptance: Float
            The acceptance rate according to which the stepsize is tuned.
            Default: 0.6
        keep: Boolean
            Whether to keep the drawn samples or discard them. Default: False
        """
        for chain in self._local_chains:
            chain.warmup(N, preferred_acceptance, keep)

    def estimate_quantity(self, function):
        """Estimates the result of a function over all samples and chains.

        Parameters
        -----------
        function: Function
        The function to be evaluated and averaged over the samples.

        Returns
        -----------
        mean, var : Tuple
        The mean and variance over the samples.

        """
        locmeanvar = [
            chain.estimate_quantity(function) for chain in self._local_chains
        ]
        locmean = [x[0] for x in locmeanvar]
        locvar = [x[1] for x in locmeanvar]
        mean = allreduce_sum(locmean, self._comm)
        var = allreduce_sum(locvar, self._comm)
        return mean/self._N_chains, var/self._N_chains

    @property
    def ESS(self):
        """The effective sample size over all samples and chains.

        Returns
        -----------
        ESS: MultiField
            The effective sample size of all model parameters.
        """
        return allreduce_sum([chain.ESS for chain in self._local_chains], self._comm)

    @property
    def R_hat(self):
        """The Gelman-Rubin test statistic R_hat.

        It measures how well the samples of different chains agree to determine
        convergence. Ideally this quantity is close to unity.

        Returns
        -----------
        R_hat: Field or MultiField
            The value of R_hat for all model parameters.
        """
        ntask, rank, master = get_MPI_params_from_comm(self._comm)
        N = len(self._local_chains[0].samples) if master else None
        if ntask > 1:
            N = self._comm.bcast(N, root=0)
        M = self._N_chains
        dom = self._dom
        locfld = [_sample_field(chain.samples) for chain in self._local_chains]
        locmeanmean = [_mean(fld, dom) for fld in locfld]
        locW = [_var(fld, dom) for fld in locfld]
        mean_mean = allreduce_sum(locmeanmean, self._comm)/M
        W = allreduce_sum(locW, self._comm)/M
        locB = [(mean_mean - _mean(fld, dom))**2 for fld in locfld]
        B = allreduce_sum(locB, self._comm)*N/(M - 1)
        var_theta = (1 - 1/N)*W + (M + 1)/(N*M)*B
        return (var_theta/W).sqrt()


class ACF_Selector(LinearOperator):
    def __init__(self, domain, N_samps):
        self._domain = DomainTuple.make(domain)
        self._N_samps = N_samps
        us_dom = UnstructuredDomain(self._N_samps//2)
        self._target = DomainTuple.make(self.domain._dom[:-1] + (us_dom,))
        self._capability = self.TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        return makeField(self._target, x.val[..., :self._N_samps//2])
