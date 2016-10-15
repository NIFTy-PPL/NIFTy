import numpy as np
cimport numpy as np
#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

cpdef np.float32_t GaussianKernel_f(np.ndarray[np.float32_t, ndim=1] mpower, np.ndarray[np.float32_t, ndim=1] mk, np.float32_t mu, np.float32_t smooth_length):
    cdef np.ndarray[np.float32_t, ndim=1] C = np.exp(-(mk - mu) ** 2 / (2. * smooth_length ** 2))
    return np.sum(C * mpower) / np.sum(C)

cpdef np.float64_t GaussianKernel(np.ndarray[np.float64_t, ndim=1] mpower, np.ndarray[np.float64_t, ndim=1] mk, np.float64_t mu, np.float64_t smooth_length):
    cdef np.ndarray[np.float64_t, ndim=1] C = np.exp(-(mk - mu) ** 2 / (2. * smooth_length ** 2))
    return np.sum(C * mpower) / np.sum(C)


cpdef np.ndarray[np.float64_t, ndim=1] apply_kernel_along_array(np.ndarray[np.float64_t, ndim=1] power, np.int_t startindex, np.int_t endindex, np.ndarray[np.float64_t, ndim=1] distances, np.float64_t smooth_length):

    if smooth_length == 0.0:
        # No smoothing requested, just return the input array.
        return power[startindex:endindex]

    #excluded_power = np.array([])
    #if (exclude > 0):
    #    distances = distances[exclude:]
    #    excluded_power = np.copy(power[:exclude])
    #    power = power[exclude:]

    if (smooth_length is None) or (smooth_length < 0):
        smooth_length = distances[1]-distances[0]

    cdef np.ndarray[np.float64_t, ndim=1] p_smooth = np.empty(endindex-startindex, dtype=np.float64)
    cdef np.int i
    for i in xrange(startindex, endindex):
        l = max(i-int(2*smooth_length)-1,0)
        u = min(i+int(2*smooth_length)+2,len(p_smooth))
        p_smooth[i-startindex] = GaussianKernel(power[l:u], distances[l:u], distances[i], smooth_length)
    #if (exclude > 0):
    #    p_smooth = np.r_[excluded_power,p_smooth]
    return p_smooth

def getShape(a):
    return tuple(a.shape)

cpdef np.ndarray[np.float64_t, ndim=1] apply_along_axis(tuple axis, np.ndarray[np.float64_t] arr, np.int_t startindex,  np.int_t endindex, np.ndarray[np.float64_t, ndim=1] distances, np.float64_t smooth_length):
    cdef tuple nd = arr.ndim
    if axis < 0:
        axis += nd
    if (axis >= nd):
        raise ValueError("axis must be less than arr.ndim; axis=%d, rank=%d."
            % (axis, nd))
    cdef np.ndarray[np.int_t, ndim=1] ind = np.zeros(nd-1)
    cdef np.ndarray[np.int_t, ndim=1] i = np.zeros(nd, 'O')
    cdef tuple shape = getShape(arr)
    cdef np.ndarray[np.int_t, ndim=1] indlist = list(range(nd))
    indlist.remove(axis)
    i[axis] = slice(None, None)
    cdef np.ndarray[np.int_t, ndim=1]  outshape = np.asarray(shape).take(indlist)
    i.put(indlist, ind)
    cdef np.ndarray[np.float64_t, ndim=1] slicedArr = arr[tuple(i.tolist())]
    cdef np.ndarray[np.float64_t, ndim=1] res = apply_kernel_along_array(slicedArr, startindex, endindex, distances, smooth_length)

    '''
    Ntot = np.product(outshape)
    holdshape = outshape
    outshape = list(arr.shape)
    outshape[axis] = len(res)
    outarr = np.zeros(outshape, np.asarray(res).dtype)
    outarr[tuple(i.tolist())] = res
    k = 1
    while k < Ntot:
        # increment the index
        ind[-1] += 1
        n = -1
        while (ind[n] >= holdshape[n]) and (n > (1-nd)):
            ind[n-1] += 1
            ind[n] = 0
            n -= 1
        i.put(indlist, ind)
        res = apply_kernel_along_array(arr[tuple(i.tolist())], startindex=startindex, endindex=endindex, distances=distances, smooth_length=smooth_length)
        outarr[tuple(i.tolist())] = res
        k += 1

    return outarr
    '''