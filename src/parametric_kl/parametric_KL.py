import nifty7 as ift
import numpy as np


m_space = ift.RGSpace([73])
a_space = ift.RGSpace([128])
b_space = ift.RGSpace([128])
o_space = ift.RGSpace([1])

mb_space = ift.MultiDomain.make({'m':[a_space,m_space],'b':[o_space,a_space]})

mb = ift.from_random(mb_space)

matop = ift.MultiLinearEinsum(mb_space,'ij,ki->jk',key_order=('m','b'))

print(matop(mb).shape)


class MultifieldFlattener(ift.LinearOperator):

    def __init__(self, domain):
        self._dof = domain.size
        self._domain = domain
        self._target = ift.DomainTuple.make(ift.UnstructuredDomain(self._dof))
        self._capability = self.TIMES | self.ADJOINT_TIMES


    def _flatten(self, x):
        result = np.empty(self.target.shape)
        runner = 0
        for key in self.domain.keys():
            dom_size = x[key].domain.size
            result[runner:runner+dom_size] = x[key].val.flatten()
            runner += dom_size
        return result

    def _restructure(self, x):
        runner = 0
        unstructured = x.val
        result = {}
        for key in self.domain.keys():
            subdom = self.domain[key]
            dom_size = subdom.size
            subfield = unstructured[runner:runner+dom_size].reshape(subdom.shape)
            subdict = {key:ift.makeField(subdom,subfield)}
            result = {**result,**subdict}
            runner += dom_size
        return result

    def apply(self, x, mode):
        self._check_mode(mode)
        if mode == self.TIMES:
            return ift.makeField(self.target,self._flatten(x))
        return ift.MultiField.from_dict(self._restructure(x))

def _get_lo_hi(comm, n_samples):
    ntask, rank, _ = ift.utilities.get_MPI_params_from_comm(comm)
    return ift.utilities.shareRange(n_samples, ntask, rank)

class ParametricGaussianKL(ift.Energy):
    """Provides the sampled Kullback-Leibler divergence between a distribution
    and a Parametric Gaussian.
    Notes
    -----

    See also

    """
    def __init__(self, variational_parameters, hamiltonian, variational_model, 
                    n_samples, mirror_samples, comm,
                        local_samples, nanisinf, _callingfrommake=False):
        if not _callingfrommake:
            raise NotImplementedError
        super(ParametricGaussianKL, self).__init__(variational_parameters)
        assert variational_model.generator.target is hamiltonian.domain
        self._hamiltonian = hamiltonian
        self._variational_model = variational_model
        self._full_model = hamiltonian(variational_model.generator) + variational_model.entropy

        self._n_samples = int(n_samples)
        self._mirror_samples = bool(mirror_samples)
        self._comm = comm
        self._local_samples = local_samples
        self._nanisinf = bool(nanisinf)

        lin = ift.Linearization.make_partial_var(variational_parameters, ['latent'])
        v, g = [], []
        for s in self._local_samples:
            # s = _modify_sample_domain(s, variational_parameters.domain)
            tmp = self._full_model(lin+s)
            tv = tmp.val.val
            tg = tmp.gradient
            if mirror_samples:
                tmp = self._full_model(lin-s)
                tv = tv + tmp.val.val
                tg = tg + tmp.gradient
            v.append(tv)
            g.append(tg)
        self._val = ift.utilities.allreduce_sum(v, self._comm)[()]/self.n_eff_samples
        if np.isnan(self._val) and self._nanisinf:
            self._val = np.inf
        self._grad = ift.utilities.allreduce_sum(g, self._comm)/self.n_eff_samples

    @staticmethod
    def make(variational_parameters, hamiltonian, variational_model, n_samples, mirror_samples,
                    comm=None, nanisinf=False):
        """Return instance of :class:`MetricGaussianKL`.

        Parameters
        ----------
        mean : Field
            Mean of the Gaussian probability distribution.
        hamiltonian : StandardHamiltonian
            Hamiltonian of the approximated probability distribution.
        n_samples : integer
            Number of samples used to stochastically estimate the KL.
        mirror_samples : boolean
            Whether the negative of the drawn samples are also used, as they are
            equally legitimate samples. If true, the number of used samples
            doubles. Mirroring samples stabilizes the KL estimate as extreme
            sample variation is counterbalanced. Since it improves stability in
            many cases, it is recommended to set `mirror_samples` to `True`.
        constants : list
            List of parameter keys that are kept constant during optimization.
            Default is no constants.
        point_estimates : list
            List of parameter keys for which no samples are drawn, but that are
            (possibly) optimized for, corresponding to point estimates of these.
            Default is to draw samples for the complete domain.
        napprox : int
            Number of samples for computing preconditioner for sampling. No
            preconditioning is done by default.
        comm : MPI communicator or None
            If not None, samples will be distributed as evenly as possible
            across this communicator. If `mirror_samples` is set, then a sample and
            its mirror image will always reside on the same task.
        nanisinf : bool
            If true, nan energies which can happen due to overflows in the forward
            model are interpreted as inf. Thereby, the code does not crash on
            these occaisions but rather the minimizer is told that the position it
            has tried is not sensible.

        Note
        ----
        The two lists `constants` and `point_estimates` are independent from each
        other. It is possible to sample along domains which are kept constant
        during minimization and vice versa.
        """

        if not isinstance(hamiltonian, ift.StandardHamiltonian):
            raise TypeError
        if hamiltonian.domain is not variational_model.generator.target:
            raise ValueError
        if not isinstance(n_samples, int):
            raise TypeError
        if not isinstance(mirror_samples, bool):
            raise TypeError
        # if isinstance(mean, MultiField) and set(point_estimates) == set(mean.keys()):
            # raise RuntimeError(
                # 'Point estimates for whole domain. Use EnergyAdapter instead.')
        n_samples = int(n_samples)
        mirror_samples = bool(mirror_samples)

        # if isinstance(variational_model.target, MultiField):
        #     cstpos = mean.extract_by_keys(point_estimates)
        #     _, ham_sampling = hamiltonian.simplify_for_constant_input(cstpos)
        # else:
        #     ham_sampling = hamiltonian
        # lin = Linearization.make_var(mean.extract(ham_sampling.domain), True)
        # met = ham_sampling(lin).metric
        # if napprox >= 1:
            # met._approximation = makeOp(approximation2endo(met, napprox))
        local_samples = []
        sseq = ift.random.spawn_sseq(n_samples)

        #FIXME dirty trick, many multiplications with zero
        DirtyMaskDict = ift.full(variational_model.generator.domain,0.).to_dict()
        DirtyMaskDict['latent'] = ift.full(variational_model.generator.domain['latent'], 1.)
        DirtyMask = ift.MultiField.from_dict(DirtyMaskDict)

        for i in range(*_get_lo_hi(comm, n_samples)):
            with ift.random.Context(sseq[i]):
                local_samples.append(DirtyMask * ift.from_random(variational_model.generator.domain))
        local_samples = tuple(local_samples)

        # if isinstance(mean, MultiField):
            # _, hamiltonian = hamiltonian.simplify_for_constant_input(mean.extract_by_keys(constants))
            # mean = mean.extract_by_keys(set(mean.keys()) - set(constants))
        return ParametricGaussianKL(
            variational_parameters, hamiltonian,variational_model,n_samples, mirror_samples, comm, local_samples,
            nanisinf, _callingfrommake=True)
    
    @property
    def n_eff_samples(self):
        if self._mirror_samples:
            return 2*self._n_samples
        return self._n_samples

    def at(self, position):
        return ParametricGaussianKL(
            position, self._hamiltonian, self._variational_model, self._n_samples, self._mirror_samples,
            self._comm, self._local_samples, self._nanisinf, True)

    @property
    def value(self):
        return self._val

    @property
    def gradient(self):
        return self._grad

class MeanfieldModel():
    def __init__(self, domain):
        self.domain = domain
        self.Flat = MultifieldFlattener(self.domain)
        self.std = ift.FieldAdapter(self.Flat.target,'var').absolute()
        self.latent = ift.FieldAdapter(self.Flat.target,'latent')
        self.mean = ift.FieldAdapter(self.Flat.target,'mean')
        self.generator = self.Flat.adjoint(self.mean + self.std * self.latent)
        self.entropy = GaussianEntropy(self.std.target) @ self.std

    def get_initial_pos(self, initial_mean=None):
        initial_pos = ift.from_random(self.generator.domain).to_dict()
        initial_pos['latent'] = ift.full(self.generator.domain['latent'], 0.)
        initial_pos['var'] = ift.full(self.generator.domain['var'], 1.)

        if initial_mean is None:
            initial_mean = 0.1*ift.from_random(self.generator.target)

        initial_pos['mean'] = self.Flat(initial_mean)
        return ift.MultiField.from_dict(initial_pos)


class GaussianEntropy(ift.EnergyOperator):

    def __init__(self, domain):

        self._domain = domain

    def apply(self, x):
        self._check_input(x)
        res =  -0.5*ift.log((2*np.pi*np.e*x**2)).sum()
        if not isinstance(x, ift.Linearization):
            return ift.Field.scalar(res)
        if not x.want_metric:
            return res
        return res.add_metric(ift.SandwichOperator.make(res.jac))
