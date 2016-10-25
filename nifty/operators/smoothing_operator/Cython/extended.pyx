import numpy as np
cimport numpy as np
#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

cpdef np.float32_t gaussianKernel_f(np.ndarray[np.float32_t, ndim=1] mpower, np.ndarray[np.float32_t, ndim=1] mk, np.float32_t mu, np.float32_t smooth_length):
    cdef np.ndarray[np.float32_t, ndim=1] C = np.exp(-(mk - mu) ** 2 / (2. * smooth_length ** 2))
    return np.sum(C * mpower) / np.sum(C)

cpdef np.float64_t gaussianKernel(np.ndarray[np.float64_t, ndim=1] mpower, np.ndarray[np.float64_t, ndim=1] mk, np.float64_t mu, np.float64_t smooth_length):
    cdef np.ndarray[np.float64_t, ndim=1] C = np.exp(-(mk - mu) ** 2 / (2. * smooth_length ** 2))
    return np.sum(C * mpower) / np.sum(C)


cpdef np.ndarray[np.float64_t, ndim=1] apply_kernel_along_array(np.ndarray[np.float64_t, ndim=1] power, np.int_t startindex, np.int_t endindex, np.ndarray[np.float64_t, ndim=1] distances, np.float64_t smooth_length):

    if smooth_length == 0.0:
        return power[startindex:endindex]

    if (smooth_length is None) or (smooth_length < 0):
        smooth_length = distances[1]-distances[0]

    cdef np.ndarray[np.float64_t, ndim=1] p_smooth = np.empty(endindex-startindex, dtype=np.float64)
    cdef np.int i
    for i in xrange(startindex, endindex):
        l = max(i-int(2*smooth_length)-1,0)
        u = min(i+int(2*smooth_length)+2,len(p_smooth))
        p_smooth[i-startindex] = gaussianKernel(power[l:u], distances[l:u], distances[i], smooth_length)

    return p_smooth

cpdef np.ndarray[np.float32_t, ndim=1] apply_kernel_along_array_f(np.ndarray[np.float32_t, ndim=1] power, np.int_t startindex, np.int_t endindex, np.ndarray[np.float32_t, ndim=1] distances, np.float32_t smooth_length):

    if smooth_length == 0.0:
        return power[startindex:endindex]

    if (smooth_length is None) or (smooth_length < 0):
        smooth_length = distances[1]-distances[0]

    cdef np.ndarray[np.float32_t, ndim=1] p_smooth = np.empty(endindex-startindex, dtype=np.float32_t)
    cdef np.int i
    for i in xrange(startindex, endindex):
        l = max(i-int(2*smooth_length)-1,0)
        u = min(i+int(2*smooth_length)+2,len(p_smooth))
        p_smooth[i-startindex] = gaussianKernel_f(power[l:u], distances[l:u], distances[i], smooth_length)

    return p_smooth


def getShape(a):
    return tuple(a.shape)

cpdef np.ndarray[np.float64_t, ndim=1] apply_along_axis(np.int_t axis, np.ndarray arr, np.int_t startindex,  np.int_t endindex, np.ndarray[np.float64_t, ndim=1] distances, np.float64_t smooth_length):
    cdef np.int_t nd = arr.ndim
    if axis < 0:
        axis += nd
    if (axis >= nd):
        raise ValueError("axis must be less than arr.ndim; axis=%d, rank=%d."
            % (axis, nd))
    cdef np.ndarray[np.int_t, ndim=1] ind = np.zeros(nd-1, dtype=np.int)
    cdef np.ndarray i = np.zeros(nd, dtype=object)
    cdef tuple shape = getShape(arr)
    cdef np.ndarray[np.int_t, ndim=1] indlist = np.asarray(range(nd))
    indlist = np.delete(indlist,axis)
    i[axis] = slice(None, None)
    cdef np.ndarray[np.int_t, ndim=1] outshape = np.asarray(shape).take(indlist)

    i.put(indlist, ind)

    cdef np.ndarray[np.float64_t, ndim=1] slicedArr
    cdef np.ndarray[np.float64_t, ndim=1] res

    cdef np.int_t Ntot = np.product(outshape)
    cdef np.ndarray[np.int_t, ndim=1] holdshape = outshape
    slicedArr = arr[tuple(i.tolist())]

    res = apply_kernel_along_array(slicedArr, startindex, endindex, distances, smooth_length)

    outshape = np.asarray(getShape(arr))
    outshape[axis] = endindex - startindex
    outarr = np.zeros(outshape, dtype=np.float64)
    outarr[tuple(i.tolist())] = res
    cdef np.int_t n, k = 1
    while k < Ntot:
        # increment the index
        ind[-1] += 1
        n = -1
        while (ind[n] >= holdshape[n]) and (n > (1-nd)):
            ind[n-1] += 1
            ind[n] = 0
            n -= 1
        i.put(indlist, ind)
        slicedArr = arr[tuple(i.tolist())]
        res = apply_kernel_along_array(slicedArr, startindex, endindex, distances, smooth_length)
        outarr[tuple(i.tolist())] = res
        k += 1

    return outarr


cpdef np.ndarray[np.float32_t, ndim=1] apply_along_axis_f(np.int_t axis, np.ndarray arr, np.int_t startindex,  np.int_t endindex, np.ndarray[np.float32_t, ndim=1] distances, np.float32_t smooth_length):
    cdef np.int_t nd = arr.ndim
    if axis < 0:
        axis += nd
    if (axis >= nd):
        raise ValueError("axis must be less than arr.ndim; axis=%d, rank=%d."
            % (axis, nd))
    cdef np.ndarray[np.int_t, ndim=1] ind = np.zeros(nd-1, dtype=np.int)
    cdef np.ndarray i = np.zeros(nd, dtype=object)
    cdef tuple shape = getShape(arr)
    cdef np.ndarray[np.int_t, ndim=1] indlist = np.asarray(range(nd))
    indlist = np.delete(indlist,axis)
    i[axis] = slice(None, None)
    cdef np.ndarray[np.int_t, ndim=1] outshape = np.asarray(shape).take(indlist)

    i.put(indlist, ind)

    cdef np.ndarray[np.float32_t, ndim=1] slicedArr
    cdef np.ndarray[np.float32_t, ndim=1] res

    cdef np.int_t Ntot = np.product(outshape)
    cdef np.ndarray[np.int_t, ndim=1] holdshape = outshape
    slicedArr = arr[tuple(i.tolist())]

    res = apply_kernel_along_array_f(slicedArr, startindex, endindex, distances, smooth_length)

    outshape = np.asarray(getShape(arr))
    outshape[axis] = endindex - startindex
    outarr = np.zeros(outshape, dtype=np.float32_t)
    outarr[tuple(i.tolist())] = res
    cdef np.int_t n, k = 1
    while k < Ntot:
        # increment the index
        ind[-1] += 1
        n = -1
        while (ind[n] >= holdshape[n]) and (n > (1-nd)):
            ind[n-1] += 1
            ind[n] = 0
            n -= 1
        i.put(indlist, ind)
        slicedArr = arr[tuple(i.tolist())]
        res = apply_kernel_along_array(slicedArr, startindex, endindex, distances, smooth_length)
        outarr[tuple(i.tolist())] = res
        k += 1

    return outarr

