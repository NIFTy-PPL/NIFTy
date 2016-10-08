import numpy as np
cimport numpy as np

ctypedef fused FUSED_TYPES_t:
    np.int32_t
    np.int64_t
    np.float32_t
    np.float64_t
    np.complex64_t
    np.complex128_t

cpdef np.ndarray[FUSED_TYPES_t, ndim=1] GaussianKernel(np.ndarray[FUSED_TYPES_t, ndim=1] mpower,np.ndarray[FUSED_TYPES_t, ndim=1] mk,np.ndarray[FUSED_TYPES_t, ndim=1] mu,np.float smooth_length):
    cdef np.ndarray[FUSED_TYPES_t, ndim=1] C = np.exp(-(mk - mu) ** 2 / (2. * smooth_length ** 2))
    return np.sum(C * mpower) / np.sum(C)


cpdef np.ndarray[FUSED_TYPES_t, ndim=1] apply_kernel_along_array(np.ndarray[FUSED_TYPES_t, ndim=1] power, np.int startindex,np.int endindex,np.ndarray[FUSED_TYPES_t, ndim=1] k,np.int exclude=1,np.float smooth_length=None):

    if smooth_length == 0:
        # No smoothing requested, just return the input array.
        return power

    cdef np.ndarray[FUSED_TYPES_t, ndim=1] excluded_power = np.array([])
    if (exclude > 0):
        k = k[exclude:]
        excluded_power = np.copy(power[:exclude])
        power = power[exclude:]

    if (smooth_length is None) or (smooth_length < 0):
        smooth_length = k[1]-k[0]

    cdef np.ndarray[FUSED_TYPES_t, ndim=1] p_smooth = np.empty(endindex-startindex)
    cdef np.ndarray[FUSED_TYPES_t, ndim=1] l,u
    cdef np.int i
    for i in xrange(startindex, endindex):
        l = max(i-int(2*smooth_length)-1,0)
        u = min(i+int(2*smooth_length)+2,len(p_smooth))
        p_smooth[i-startindex] = GaussianKernel(power[l:u], k[l:u], k[i],
                                          smooth_length)

    if (exclude > 0):
        p_smooth = np.r_[excluded_power,p_smooth]
    return p_smooth


cpdef np.ndarray[FUSED_TYPES_t] apply_along_axis(tuple axis,np.ndarray[FUSED_TYPES_t] arr, np.int startindex,np.int endindex,np.ndarray[FUSED_TYPES_t, ndim=1] distances,np.int exclude=1,np.float smooth_length=None):
    cdef tuple nd = arr.ndim
    if axis < 0:
        axis += nd
    if (axis >= nd):
        raise ValueError("axis must be less than arr.ndim; axis=%d, rank=%d."
            % (axis, nd))
    cdef np.ndarray ind = np.zeros(nd-1)
    cdef np.ndarray i = np.zeros(nd, 'O')
    indlist = list(range(nd))
    indlist.remove(axis)
    i[axis] = slice(None, None)
    cdef np.ndarray outshape = np.asarray(arr.shape).take(indlist)
    i.put(indlist, ind)
    cdef np.ndarray res = apply_kernel_along_array(arr[tuple(i.tolist())], startindex=startindex,endindex=endindex,k=distances,exclude=exclude,smooth_length=smooth_length)

    cdef np.int Ntot = np.product(outshape)
    holdshape = outshape
    outshape = list(arr.shape)
    outshape[axis] = len(res)
    outarr = np.zeros(outshape, np.asarray(res).dtype)
    outarr[tuple(i.tolist())] = res
    cdef np.int k = 1
    cdef np.int n
    while k < Ntot:
        # increment the index
        ind[-1] += 1
        n = -1
        while (ind[n] >= holdshape[n]) and (n > (1-nd)):
            ind[n-1] += 1
            ind[n] = 0
            n -= 1
        i.put(indlist, ind)
        res = apply_kernel_along_array(arr[tuple(i.tolist())], startindex=startindex,endindex=endindex,k=distances, exclude=exclude,smooth_length=smooth_length)
        outarr[tuple(i.tolist())] = res
        k += 1
    return outarr