import numpy as np
from ..operators.multifield_flattener import MultifieldFlattener
from ..operators.simple_linear_operators import FieldAdapter, PartialExtractor
from ..operators.energy_operators import EnergyOperator
from ..operators.sandwich_operator import SandwichOperator
from ..operators.linear_operator import LinearOperator
from ..operators.einsum import MultiLinearEinsum
from ..sugar import full, from_random, makeField, domain_union
from ..linearization import Linearization
from ..field import Field
from ..multi_field import MultiField
from ..domain_tuple import DomainTuple
from ..domains.unstructured_domain import UnstructuredDomain

class MeanfieldModel():
    def __init__(self, domain):
        self.domain = domain
        self.Flat = MultifieldFlattener(self.domain)

        self.std = FieldAdapter(self.Flat.target,'var').absolute()
        self.latent = FieldAdapter(self.Flat.target,'latent')
        self.mean = FieldAdapter(self.Flat.target,'mean')
        self.generator = self.Flat.adjoint(self.mean + self.std * self.latent)
        self.entropy = GaussianEntropy(self.std.target) @ self.std

    def get_initial_pos(self, initial_mean=None):
        initial_pos = from_random(self.generator.domain).to_dict()
        initial_pos['latent'] = full(self.generator.domain['latent'], 0.)
        initial_pos['var'] = full(self.generator.domain['var'], 1.)

        if initial_mean is None:
            initial_mean = 0.1*from_random(self.generator.target)

        initial_pos['mean'] = self.Flat(initial_mean)
        return MultiField.from_dict(initial_pos)

class FullCovarianceModel():
    def __init__(self, domain):
        self.domain = domain
        self.Flat = MultifieldFlattener(self.domain)
        one_space = UnstructuredDomain(1)
        self.flat_domain = self.Flat.target[0]
        N_tri = self.flat_domain.shape[0]*(self.flat_domain.shape[0]+1)//2
        triangular_space = DomainTuple.make(UnstructuredDomain(N_tri))
        tri = FieldAdapter(triangular_space, 'cov')
        mat_space = DomainTuple.make((self.flat_domain,self.flat_domain))
        lat_mat_space = DomainTuple.make((one_space,self.flat_domain))
        lat = FieldAdapter(lat_mat_space,'latent')
        LT = LowerTriangularProjector(triangular_space,mat_space)
        mean = FieldAdapter(self.flat_domain,'mean')
        cov = LT @ tri
        co = FieldAdapter(cov.target, 'co')

        matmul_setup_dom = domain_union((co.domain,lat.domain))
        co_part = PartialExtractor(matmul_setup_dom, co.domain)
        lat_part = PartialExtractor(matmul_setup_dom, lat.domain)
        matmul_setup = lat_part.adjoint @ lat.adjoint @ lat + co_part.adjoint @ co.adjoint @ cov
        MatMult = MultiLinearEinsum(matmul_setup.target,'ij,ki->jk', key_order=('co','latent'))

        Resp = Respacer(MatMult.target, mean.target)
        self.generator = self.Flat.adjoint @ (mean + Resp @ MatMult @ matmul_setup)
        
        Diag = DiagonalSelector(cov.target, self.Flat.target)
        diag_cov = Diag(cov).absolute()
        self.entropy = GaussianEntropy(diag_cov.target) @ diag_cov

    def get_initial_pos(self, initial_mean = None):
        initial_pos = from_random(self.generator.domain).to_dict()
        initial_pos['latent'] = full(self.generator.domain['latent'], 0.)
        diag_tri = np.diag(np.ones(self.flat_domain.shape[0]))[np.tril_indices(self.flat_domain.shape[0])]
        initial_pos['cov'] = makeField(self.generator.domain['cov'], diag_tri)
        if initial_mean is None:
            initial_mean = 0.1*from_random(self.generator.target)
        initial_pos['mean'] = self.Flat(initial_mean)
        return MultiField.from_dict(initial_pos)



class GaussianEntropy(EnergyOperator):
    def __init__(self, domain):
        self._domain = domain

    def apply(self, x):
        self._check_input(x)
        res =  -0.5* (2*np.pi*np.e*x**2).log().sum()
        if not isinstance(x, Linearization):
            return Field.scalar(res)
        if not x.want_metric:
            return res
        return res.add_metric(SandwichOperator.make(res.jac)) #FIXME not sure about metric


class LowerTriangularProjector(LinearOperator):
    def __init__(self, domain, target):
        self._domain = domain
        self._target = target
        self._indices=np.tril_indices(target.shape[0])
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_mode(mode)
        if mode == self.TIMES:
            mat = np.zeros(self._target.shape)
            mat[self._indices] = x.val
            return makeField(self._target,mat)
        return makeField(self._domain, x.val[self._indices].reshape(self._domain.shape))

class DiagonalSelector(LinearOperator):
    def __init__(self, domain, target):
        self._domain = domain
        self._target = target
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_mode(mode)
        if mode == self.TIMES:
            result = np.diag(x.val)
            return makeField(self._target,result)
        return makeField(self._domain,np.diag(x.val).reshape(self._domain.shape))


class Respacer(LinearOperator):
    def __init__(self, domain, target):
        self._domain = domain
        self._target = target
        self._capability = self.TIMES | self.ADJOINT_TIMES


    def apply(self,x,mode):
        self._check_mode(mode)
        if mode == self.TIMES:
            return makeField(self._target,x.val.reshape(self._target.shape))
        return makeField(self._domain,x.val.reshape(self._domain.shape))
