import numpy as np
cimport numpy as np

cpdef _distance_array_helper(np.ndarray[np.int_t] index_array, np.int_t lmax):
    cdef np.int size = (lmax + 1) * (lmax  + 1)

    cdef np.ndarray res = np.zeros([len(index_array)], dtype=np.float128)

    for idx, index_full in enumerate(index_array):
        if index_full <= lmax:
            index_half = index_full
        else:
            index_half = (index_full + lmax + 1) / 2;
        m = (np.ceil(((2 * lmax + 1) -
                      np.sqrt((2 * lmax + 1)**2 -
                              8 * (index_half - lmax))) / 2)).astype(int)
        res[idx] = index_half - m * (2 * lmax + 1 - m) // 2

    return res
