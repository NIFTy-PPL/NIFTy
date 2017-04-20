
import numpy as np
cimport numpy as np

def buildLm(inp, **kwargs):
    if inp.dtype == np.dtype('float32'):
        return _buildLm_f(inp, **kwargs)
    else:
        return _buildLm(inp, **kwargs)

def buildIdx(inp, **kwargs):
    if inp.dtype == np.dtype('complex64'):
        return _buildIdx_f(inp, **kwargs)
    else:
        return _buildIdx(inp, **kwargs)

cpdef np.ndarray[np.float32_t, ndim=1]  _buildIdx_f(np.ndarray[np.complex64_t,
ndim=1] nr, np.int_t lmax):
    cdef np.int size = (lmax+1)*(lmax+1)

    cdef np.ndarray final=np.zeros([size], dtype=np.float32)
    final[0:lmax+1] = nr[0:lmax+1].real
    final[lmax+1::2] = np.sqrt(2)*nr[lmax+1:].real
    final[lmax+2::2] = np.sqrt(2)*nr[lmax+1:].imag
    return final

cpdef np.ndarray[np.float64_t, ndim=1]  _buildIdx(np.ndarray[np.complex128_t,
ndim=1] nr, np.int_t lmax):
    cdef np.int size = (lmax+1)*(lmax+1)

    cdef np.ndarray final=np.zeros([size], dtype=np.float64)
    final[0:lmax+1] = nr[0:lmax+1].real
    final[lmax+1::2] = np.sqrt(2)*nr[lmax+1:].real
    final[lmax+2::2] = np.sqrt(2)*nr[lmax+1:].imag
    return final

cpdef np.ndarray[np.complex64_t, ndim=1] _buildLm_f(np.ndarray[np.float32_t,
ndim=1] nr, np.int_t lmax):
    cdef np.int size = (len(nr)-lmax-1)/2+lmax+1

    cdef np.ndarray res=np.zeros([size], dtype=np.complex64)
    res[0:lmax+1] = nr[0:lmax+1]
    res[lmax+1:] = np.sqrt(0.5)*(nr[lmax+1::2] + 1j*nr[lmax+2::2])
    return res

cpdef np.ndarray[np.complex128_t, ndim=1] _buildLm(np.ndarray[np.float64_t,
ndim=1] nr, np.int_t lmax):
    cdef np.int size = (len(nr)-lmax-1)/2+lmax+1

    cdef np.ndarray res=np.zeros([size], dtype=np.complex128)
    res[0:lmax+1] = nr[0:lmax+1]
    res[lmax+1:] = np.sqrt(0.5)*(nr[lmax+1::2] + 1j*nr[lmax+2::2])
    return res
