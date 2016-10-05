import numpy as np

def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    Apply a function to 1-D slices along the given axis.

    Execute `func1d(a, *args)` where `func1d` operates on 1-D arrays and `a`
    is a 1-D slice of `arr` along `axis`.

    Parameters
    ----------
    func1d : function
        This function should accept 1-D arrays. It is applied to 1-D
        slices of `arr` along the specified axis.
    axis : integer
        Axis along which `arr` is sliced.
    arr : ndarray
        Input array.
    args : any
        Additional arguments to `func1d`.
    kwargs: any
        Additional named arguments to `func1d`.

        .. versionadded:: 1.9.0


    Returns
    -------
    apply_along_axis : ndarray
        The output array. The shape of `outarr` is identical to the shape of
        `arr`, except along the `axis` dimension, where the length of `outarr`
        is equal to the size of the return value of `func1d`.  If `func1d`
        returns a scalar `outarr` will have one fewer dimensions than `arr`.
    """
    arr = np.asarray(arr)
    nd = arr.ndim
    if axis < 0:
        axis += nd
    if (axis >= nd):
        raise ValueError("axis must be less than arr.ndim; axis=%d, rank=%d."
            % (axis, nd))
    ind = [0]*(nd-1)
    i = np.zeros(nd, 'O')
    indlist = list(range(nd))
    indlist.remove(axis)
    i[axis] = slice(None, None)
    outshape = np.asarray(arr.shape).take(indlist)
    i.put(indlist, ind)
    res = func1d(arr[tuple(i.tolist())], *args, **kwargs)

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
        res = func1d(arr[tuple(i.tolist())], *args, **kwargs)
        outarr[tuple(i.tolist())] = res
        k += 1
    return outarr