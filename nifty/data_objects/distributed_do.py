import numpy as np
from .random import Random
from mpi4py import MPI

comm = MPI.COMM_WORLD
ntask = comm.Get_size()
rank = comm.Get_rank()


def shareSize(nwork, nshares, myshare):
    nbase = nwork//nshares
    return nbase if myshare>=nwork%nshares else nbase+1
def shareRange(nwork, nshares, myshare):
    nbase = nwork//nshares;
    additional = nwork%nshares;
    lo = myshare*nbase + min(myshare, additional)
    hi = lo+nbase+ (1 if myshare<additional else 0)
    return lo,hi

def local_shape(shape, distaxis):
    if len(shape)==0:
        distaxis = -1
    if distaxis==-1:
        return shape
    shape2=list(shape)
    shape2[distaxis]=shareSize(shape[distaxis],ntask,rank)
    return tuple(shape2)

class data_object(object):
    def __init__(self, shape, data, distaxis):
        """Must not be called directly by users"""
        self._shape = tuple(shape)
        if len(self._shape)==0:
            distaxis = -1
        self._distaxis = distaxis
        lshape = local_shape(self._shape, self._distaxis)
        self._data = data

    def sanity_checks(self):
        # check whether the distaxis is consistent
        if self._distaxis<-1 or self._distaxis>=len(self._shape):
            raise ValueError
        itmp=np.array(self._distaxis)
        otmp=np.empty(ntask,dtype=np.int)
        comm.Allgather(itmp,otmp)
        if np.any(otmp!=self._distaxis):
            raise ValueError
        # check whether the global shape is consistent
        itmp=np.array(self._shape)
        otmp=np.empty((ntask,len(self._shape)),dtype=np.int)
        comm.Allgather(itmp,otmp)
        for i in range(ntask):
            if np.any(otmp[i,:]!=self._shape):
                raise ValueError
        # check shape of local data
        if self._distaxis<0:
            if self._data.shape!=self._shape:
                raise ValueError
        else:
            itmp=np.array(self._shape)
            itmp[self._distaxis] = shareSize(self._shape[self._distaxis],ntask,rank)
            if np.any(self._data.shape!=itmp):
                raise ValueError

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return np.prod(self._shape)

    @property
    def real(self):
        return data_object(self._shape, self._data.real, self._distaxis)

    @property
    def imag(self):
        return data_object(self._shape, self._data.imag, self._distaxis)

    def _contraction_helper(self, op, mpiop, axis):
        if axis is not None:
            if len(axis)==len(self._data.shape):
                axis = None
        if axis is None:
            res = np.array(getattr(self._data, op)())
            if (self._distaxis==-1):
                return res[0]
            res2 = np.empty(1,dtype=res.dtype)
            comm.Allreduce(res,res2,mpiop)
            return res2[0]

        if self._distaxis in axis:
            res = getattr(self._data, op)(axis=axis)
            res2 = np.empty_like(res)
            comm.Allreduce(res,res2,mpiop)
            return from_global_data(res2, distaxis=0)
        else:
            # perform the contraction on the local data
            res = getattr(self._data, op)(axis=axis)
            if self._distaxis == -1:
                return from_global_data(res,distaxis=0)
            shp = list(res.shape)
            shift=0
            for ax in axis:
                if ax<self._distaxis:
                    shift+=1
            shp[self._distaxis-shift] = self.shape[self._distaxis]
            return from_local_data(shp, res, self._distaxis-shift)

        # check if the result is scalar or if a result_field must be constr.
        if np.isscalar(data):
            return data
        else:
            return data_object(data)

    def sum(self, axis=None):
        return self._contraction_helper("sum", MPI.SUM, axis)

    # FIXME: to be improved!
    def mean(self):
        return self.sum()/self.size
    def std(self):
        return np.sqrt(self.var())
    def var(self):
        return (abs(self-self.mean())**2).mean()

    def _binary_helper(self, other, op):
        a = self
        if isinstance(other, data_object):
            b = other
            if a._shape != b._shape:
                raise ValueError("shapes are incompatible.")
            if a._distaxis != b._distaxis:
                raise ValueError("distributions are incompatible.")
            a = a._data
            b = b._data
        else:
            a = a._data
            b = other

        tval = getattr(a, op)(b)
        return self if tval is a else data_object(self._shape, tval, self._distaxis)

    def __add__(self, other):
        return self._binary_helper(other, op='__add__')

    def __radd__(self, other):
        return self._binary_helper(other, op='__radd__')

    def __iadd__(self, other):
        return self._binary_helper(other, op='__iadd__')

    def __sub__(self, other):
        return self._binary_helper(other, op='__sub__')

    def __rsub__(self, other):
        return self._binary_helper(other, op='__rsub__')

    def __isub__(self, other):
        return self._binary_helper(other, op='__isub__')

    def __mul__(self, other):
        return self._binary_helper(other, op='__mul__')

    def __rmul__(self, other):
        return self._binary_helper(other, op='__rmul__')

    def __imul__(self, other):
        return self._binary_helper(other, op='__imul__')

    def __div__(self, other):
        return self._binary_helper(other, op='__div__')

    def __rdiv__(self, other):
        return self._binary_helper(other, op='__rdiv__')

    def __idiv__(self, other):
        return self._binary_helper(other, op='__idiv__')

    def __truediv__(self, other):
        return self._binary_helper(other, op='__truediv__')

    def __rtruediv__(self, other):
        return self._binary_helper(other, op='__rtruediv__')

    def __pow__(self, other):
        return self._binary_helper(other, op='__pow__')

    def __rpow__(self, other):
        return self._binary_helper(other, op='__rpow__')

    def __ipow__(self, other):
        return self._binary_helper(other, op='__ipow__')

    def __eq__(self, other):
        return self._binary_helper(other, op='__eq__')

    def __ne__(self, other):
        return self._binary_helper(other, op='__ne__')

    def __neg__(self):
        return data_object(self._shape,-self._data,self._distaxis)

    def __abs__(self):
        return data_object(self._shape,np.abs(self._data),self._distaxis)

    #def ravel(self):
    #    return data_object(self._data.ravel())

    #def reshape(self, shape):
    #    return data_object(self._data.reshape(shape))

    def all(self):
        return self._data.all()

    def any(self):
        return self._data.any()


def full(shape, fill_value, dtype=None, distaxis=0):
    return data_object(shape, np.full(local_shape(shape, distaxis), fill_value, dtype), distaxis)


def empty(shape, dtype=None, distaxis=0):
    return data_object(shape, np.empty(local_shape(shape, distaxis), dtype), distaxis)


def zeros(shape, dtype=None, distaxis=0):
    return data_object(shape, np.zeros(local_shape(shape, distaxis), dtype), distaxis)


def ones(shape, dtype=None, distaxis=0):
    return data_object(shape, np.ones(local_shape(shape, distaxis), dtype), distaxis)


def empty_like(a, dtype=None):
    return data_object(np.empty_like(a._data, dtype))


def vdot(a, b):
    tmp = np.vdot(a._data.ravel(), b._data.ravel())
    res = np.empty(1,dtype=type(tmp))
    comm.Allreduce(tmp,res,MPI.SUM)
    return res[0]


def _math_helper(x, function, out):
    if out is not None:
        function(x._data, out=out._data)
        return out
    else:
        return data_object(x.shape,function(x._data),x._distaxis)


def abs(a, out=None):
    return _math_helper(a, np.abs, out)


def exp(a, out=None):
    return _math_helper(a, np.exp, out)


def log(a, out=None):
    return _math_helper(a, np.log, out)


def sqrt(a, out=None):
    return _math_helper(a, np.sqrt, out)


def bincount(x, weights=None, minlength=None):
    if weights is not None:
        weights = weights._data
    res = np.bincount(x._data, weights, minlength)
    return data_object(res)


def from_object(object, dtype=None, copy=True):
    return data_object(object._shape, np.array(object._data, dtype=dtype, copy=copy), distaxis=object._distaxis)


def from_random(random_type, shape, dtype=np.float64, distaxis=0, **kwargs):
    generator_function = getattr(Random, random_type)
    #lshape = local_shape(shape, distaxis)
    #return data_object(shape, generator_function(dtype=dtype, shape=lshape, **kwargs), distaxis=distaxis)
    return from_global_data(generator_function(dtype=dtype, shape=shape, **kwargs), distaxis=distaxis)

def local_data(arr):
    return arr._data


def ibegin(arr):
    res = [0] * arr._data.ndim
    res[arr._distaxis] = shareRange(arr._shape[arr._distaxis],ntask,rank)[0]
    return tuple(res)


def np_allreduce_sum(arr):
    res = np.empty_like(arr)
    comm.Allreduce(arr,res,MPI.SUM)
    return res


def distaxis(arr):
    return arr._distaxis


def from_local_data (shape, arr, distaxis):
    return data_object(shape, arr, distaxis)


def from_global_data (arr, distaxis=0):
    if distaxis==-1:
        return data_object(arr.shape, arr, distaxis)
    lo, hi = shareRange(arr.shape[distaxis],ntask,rank)
    sl = [slice(None)]*len(arr.shape)
    sl[distaxis]=slice(lo,hi)
    return data_object(arr.shape, arr[sl], distaxis)


def to_global_data (arr):
    if arr._distaxis==-1:
        return arr._data
    tmp = redistribute(arr, dist=-1)
    return tmp._data


def redistribute (arr, dist=None, nodist=None):
    if dist is not None:
        if nodist is not None:
            raise ValueError
        if dist==arr._distaxis:
            return arr
    else:
        if nodist is None:
            raise ValueError
        if arr._distaxis not in nodist:
            return arr
        dist=-1
        for i in range(len(arr.shape)):
            if i not in nodist:
                dist=i
                break

    if arr._distaxis==-1:  # just pick the proper subset
        return from_global_data(arr._data, dist)
    if dist==-1: # gather data
        tmp = np.moveaxis(arr._data, arr._distaxis, 0)
        slabsize=np.prod(tmp.shape[1:])*tmp.itemsize
        sz=np.empty(ntask,dtype=np.int)
        for i in range(ntask):
            sz[i]=slabsize*shareSize(arr.shape[arr._distaxis],ntask,i)
        disp=np.empty(ntask,dtype=np.int)
        disp[0]=0
        disp[1:]=np.cumsum(sz[:-1])
        tmp=tmp.flatten()
        out = np.empty(arr.size,dtype=arr.dtype)
        comm.Allgatherv(tmp,[out,sz,disp,MPI.BYTE])
        shp = np.array(arr._shape)
        shp[1:arr._distaxis+1] = shp[0:arr._distaxis]
        shp[0] = arr.shape[arr._distaxis]
        out = out.reshape(shp)
        out = np.moveaxis(out, 0, arr._distaxis)
        return from_global_data (out, distaxis=-1)
    # real redistribution via Alltoallv
    # temporary slow, but simple solution
    #return redistribute(redistribute(arr,dist=-1),dist=dist)

    tmp = np.moveaxis(arr._data, (dist, arr._distaxis), (0, 1))
    tshape = tmp.shape
    slabsize=np.prod(tmp.shape[2:])*tmp.itemsize
    ssz=np.empty(ntask,dtype=np.int)
    rsz=np.empty(ntask,dtype=np.int)
    for i in range(ntask):
        ssz[i]=shareSize(arr.shape[dist],ntask,i)*tmp.shape[1]*slabsize
        rsz[i]=shareSize(arr.shape[dist],ntask,rank)*shareSize(arr.shape[arr._distaxis],ntask,i)*slabsize
    sdisp=np.empty(ntask,dtype=np.int)
    rdisp=np.empty(ntask,dtype=np.int)
    sdisp[0]=0
    rdisp[0]=0
    sdisp[1:]=np.cumsum(ssz[:-1])
    rdisp[1:]=np.cumsum(rsz[:-1])
    tmp=tmp.flatten()
    out = np.empty(np.prod(local_shape(arr.shape,dist)),dtype=arr.dtype)
    s_msg = [tmp, (ssz, sdisp), MPI.BYTE]
    r_msg = [out, (rsz, rdisp), MPI.BYTE]
    comm.Alltoallv(s_msg, r_msg)
    out2 = np.empty([shareSize(arr.shape[dist],ntask,rank), arr.shape[arr._distaxis]] +list(tshape[2:]), dtype=arr.dtype)
    ofs=0
    for i in range(ntask):
        lsize = rsz[i]//tmp.itemsize
        lo,hi = shareRange(arr.shape[arr._distaxis],ntask,i)
        out2[slice(None),slice(lo,hi)] = out[ofs:ofs+lsize].reshape([shareSize(arr.shape[dist],ntask,rank),shareSize(arr.shape[arr._distaxis],ntask,i)]+list(tshape[2:]))
        ofs += lsize
    new_shape = [shareSize(arr.shape[dist],ntask,rank), arr.shape[arr._distaxis]] +list(tshape[2:])
    out2=out2.reshape(new_shape)
    out2 = np.moveaxis(out2, (0, 1), (dist, arr._distaxis))
    return from_local_data (arr.shape, out2, dist)


def default_distaxis():
    return 0
