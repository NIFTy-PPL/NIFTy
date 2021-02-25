import nifty7 as ift
import numpy as np
import os, sys



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
            subdict = {key:ift.from_global_data(subdom,subfield)}
            result = {**result,**subdict}
            runner += dom_size
        return result

    def apply(self, x, mode):
        self._check_mode(mode)
        if mode == self.TIMES:
            return ift.from_global_data(self.target,self._flatten(x))
        return ift.MultiField.from_dict(self._restructure(x))

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


class LowerTriangularProjector(ift.LinearOperator):
    def __init__(self, domain, target):
        self._domain = domain
        self._target = target
        self._indices=np.tril_indices(target.shape[1])
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_mode(mode)
        if mode == self.TIMES:
            mat = np.zeros(self._target.shape[1:])
            mat[self._indices] = x.val
            return ift.from_global_data(self._target,mat.reshape((1,)+mat.shape))
        return ift.from_global_data(self._domain,x.val[0][self._indices].reshape(self._domain.shape))

class DiagonalSelector(ift.LinearOperator):
    def __init__(self, domain, target):
        self._domain = domain
        self._target = target
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_mode(mode)
        if mode == self.TIMES:
            result = np.diag(x.val[0])
            return ift.from_global_data(self._target,result)
        return ift.from_global_data(self._domain,np.diag(x.val).reshape(self._domain.shape))



class ParametrizedGaussianKL(ift.Energy):

    def __init__(self, variational_parameters, hamiltonian, variational_model, entropy ,n_samples,
                 _samples=None):
        super(ParametrizedGaussianKL, self).__init__(variational_parameters)
#       \xi =  \bar{\xi} + \sigma * \eta
        if hamiltonian.domain is not variational_model.target:
            raise ValueError
        if not isinstance(n_samples, int):
            raise TypeError
        self._entropy = entropy
        self._n_samples = n_samples
        self._hamiltonian = hamiltonian
        self._variational_model = variational_model
        self._full_model = hamiltonian(variational_model) + entropy
        #FIXME !DIRTY, DON'T TO THIS!
        DirtyMaskDict = ift.full(self._variational_model.domain,0.).to_dict()
        DirtyMaskDict['latent'] = ift.full(self._variational_model.domain['latent'], 1.)
        DirtyMask = ift.MultiField.from_dict(DirtyMaskDict)

        if _samples is None:

            _samples = tuple(DirtyMask * ift.from_random('normal', variational_model.domain)
                             for _ in range(n_samples))
        else:

            _samples = tuple(DirtyMask * _
                             for _ in _samples)

        ##################

        self._samples = _samples

        self._lin = ift.Linearization.make_partial_var(variational_parameters, ['latent'])
        v, g = None, None
        for s in self._samples:
            tmp = self._full_model(self._lin+s)
            if v is None:
                v = tmp.val.local_data[()]
                g = tmp.gradient
            else:
                v += tmp.val.local_data[()]
                g = g + tmp.gradient
        self._val = v / len(self._samples)
        self._grad = g * (1./len(self._samples))
        self._metric = None

    def at(self, position):
        return ParametrizedGaussianKL(position, self._hamiltonian, self._variational_model,
                                      self._entropy, n_samples=self._n_samples,
                                      _samples=self._samples)

    @property
    def value(self):
        return self._val

    @property
    def gradient(self):
        return self._grad

    def _get_metric(self):
        if self._metric is None:
            lin = self._lin.with_want_metric()
            mymap = map(lambda v: self._full_model(lin+v).metric,
                        self._samples)
            self._metric = ift.utilities.my_sum(mymap)
            self._metric = self._metric.scale(1./len(self._samples))

    def apply_metric(self, x):
        self._get_metric()
        return self._metric(x)

    @property
    def metric(self):
        self._get_metric()
        return self._metric

    @property
    def samples(self):
        return self._samples

class Respacer(ift.LinearOperator):
    def __init__(self, domain, target):
        self._domain = domain
        self._target = target
        self._capability = self.TIMES | self.ADJOINT_TIMES


    def apply(self,x,mode):
        self._check_mode(mode)
        if mode == self.TIMES:
            return ift.from_global_data(self._target,x.val.reshape(self._target.shape))
        return ift.from_global_data(self._domain,x.val.reshape(self._domain.shape))

def build_meanfield(likelihood, initial_mean =None):

    Flat = MultifieldFlattener(likelihood.domain)
    variational_var = ift.FieldAdapter(Flat.target,'var').absolute()
    variational_latent = ift.FieldAdapter(Flat.target,'latent')
    variational_mean = ift.FieldAdapter(Flat.target,'mean')
    meanfield_model = Flat.adjoint(variational_mean + variational_var * variational_latent)
    initial_pos = ift.from_random('normal', meanfield_model.domain).to_dict()
    initial_pos['latent'] = ift.full(meanfield_model.domain['latent'], 0.)
    initial_pos['var'] = ift.full(meanfield_model.domain['var'], 1.)

    if initial_mean is None:
        initial_mean = 0.1*ift.from_random('normal',likelihood.domain)

    initial_pos['mean'] = Flat(initial_mean)

    initial_pos = ift.MultiField.from_dict(initial_pos)

    meanfield_entropy = GaussianEntropy(variational_var.target)(variational_var)

    return meanfield_model, meanfield_entropy, initial_pos, variational_var, variational_mean

def build_fullcovariance(likelihood, initial_mean =None):

    Flat = MultifieldFlattener(likelihood.domain)

    one_space = ift.UnstructuredDomain(1)
    latent_domain = Flat.target[0]
    N_tri = latent_domain.shape[0]*(latent_domain.shape[0]+1)//2
    triangular_space = ift.DomainTuple.make(ift.UnstructuredDomain(N_tri))
    tri = ift.FieldAdapter(triangular_space,'cov')
    mat_space = ift.DomainTuple.make((one_space,latent_domain,latent_domain))
    lat_mat_space = ift.DomainTuple.make((one_space,one_space,latent_domain))
    lat = ift.FieldAdapter(lat_mat_space,'latent')
    LT = LowerTriangularProjector(triangular_space,mat_space)
    mea = ift.FieldAdapter(latent_domain,'mea')
    cov = LT @ tri
    Mmult = VariableMatMul(lat,cov)
    Resp = Respacer(Mmult.target,mea.target)
    sam = Resp(Mmult) + mea

    Diag = DiagonalSelector(cov.target,Flat.target)

    fullcovariance_model = Flat.adjoint(sam)

    Diag_cov = Diag(cov).absolute()
    fullcovariance_entropy = GaussianEntropy(Diag_cov.target)(Diag(cov))
    initial_pos = ift.from_random('normal', fullcovariance_model.domain).to_dict()
    initial_pos['latent'] = ift.full(fullcovariance_model.domain['latent'], 0.)
    diag_tri = np.diag(np.ones(latent_domain.shape[0]))[np.tril_indices(latent_domain.shape[0])]
    initial_pos['cov'] = ift.from_global_data(fullcovariance_model.domain['cov'], diag_tri)
    initial_pos['mea'] = Flat(initial_mean)
    # return initial_pos
    initial_pos = ift.MultiField.from_dict(initial_pos)
    return  fullcovariance_model, fullcovariance_entropy, initial_pos, cov, mea
