import numpy as np


def buildLm(nr, lmax):
    new_dtype = np.result_type(nr.dtype, np.complex64)

    size = (len(nr)-lmax-1)/2+lmax+1
    res = np.zeros([size], dtype=new_dtype)
    res[0:lmax+1] = nr[0:lmax+1]
    res[lmax+1:] = np.sqrt(0.5)*(nr[lmax+1::2] + 1j*nr[lmax+2::2])
    return res


def buildIdx(nr, lmax):
    if nr.dtype == np.dtype('complex64'):
        new_dtype = np.float32
    elif nr.dtype == np.dtype('complex128'):
        new_dtype = np.float64
    elif nr.dtype == np.dtype('complex256'):
        new_dtype = np.float128
    else:
        raise TypeError("dtype of nr not supported.")

    size = (lmax+1)*(lmax+1)
    final = np.zeros([size], dtype=new_dtype)
    final[0:lmax+1] = nr[0:lmax+1].real
    final[lmax+1::2] = np.sqrt(2)*nr[lmax+1:].real
    final[lmax+2::2] = np.sqrt(2)*nr[lmax+1:].imag
    return final
