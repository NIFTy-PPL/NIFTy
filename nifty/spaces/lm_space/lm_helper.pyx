import numpy as np
cimport numpy as np

cpdef _distance_array_helper(np.int_t x, np.ndarray[np.int_t] l,
                             np.int_t original_size, np.int_t lmax):
    cdef np.int size = (lmax + 1) * (lmax  + 1)

    cdef np.ndarray res = np.zeros([size], dtype=np.float128)
    res[0:lmax + 1] = np.arange(lmax + 1)

    for i in xrange(original_size - lmax - 1):
        res[i * 2 + lmax + 1] = l[i + lmax + 1]
        res[i * 2 + lmax + 1] = l[i + lmax + 1]

    return res[x]
