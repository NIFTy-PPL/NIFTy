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

cpdef np.ndarray[np.float32_t, ndim=1]  _buildIdx_f(np.ndarray[np
.complex64_t, ndim=1] nr, np.int_t lmax):
    cdef np.int size = (lmax+1)*(lmax+1)

    cdef np.ndarray res=np.zeros([size], dtype=np.complex64)
    cdef np.ndarray final=np.zeros([size], dtype=np.float32)
    res[0:lmax+1] = nr[0:lmax+1]

    for i in xrange(len(nr)-lmax-1):
        res[i*2+lmax+1] = nr[i+lmax+1]
        res[i*2+lmax+2] = np.conjugate(nr[i+lmax+1])
    final = _realify_f(res, lmax)
    return final

cpdef np.ndarray[np.float32_t, ndim=1] _realify_f(np.ndarray[np.complex64_t, ndim=1] nr, np.int_t lmax):
    cdef np.ndarray resi=np.zeros([len(nr)], dtype=np.float32)

    resi[0:lmax+1] = np.real(nr[0:lmax+1])

    for i in xrange(lmax+1,len(nr),2):
        mi =  int(np.ceil(((2*lmax+1)-np.sqrt((2*lmax+1)*(2*lmax+1)-4*(i-lmax)+1))/2))
        resi[i]=np.sqrt(2)*np.real(nr[i])*(-1)**(mi*mi)
        resi[i+1]=np.sqrt(2)*np.imag(nr[i])*(-1)**(mi*mi)
    return resi

cpdef np.ndarray[np.float64_t, ndim=1]  _buildIdx(np.ndarray[np.complex128_t,
 ndim=1] nr, np.int_t lmax):
    cdef np.int size = (lmax+1)*(lmax+1)

    cdef np.ndarray res=np.zeros([size], dtype=np.complex128)
    cdef np.ndarray final=np.zeros([size], dtype=np.float64)
    res[0:lmax+1] = nr[0:lmax+1]

    for i in xrange(len(nr)-lmax-1):
        res[i*2+lmax+1] = nr[i+lmax+1]
        res[i*2+lmax+2] = np.conjugate(nr[i+lmax+1])
    final = _realify(res, lmax)
    return final

cpdef np.ndarray[np.float64_t, ndim=1] _realify(np.ndarray[np.complex128_t, ndim=1] nr, np.int_t lmax):
    cdef np.ndarray resi=np.zeros([len(nr)], dtype=np.float64)

    resi[0:lmax+1] = np.real(nr[0:lmax+1])

    for i in xrange(lmax+1,len(nr),2):
        mi =  int(np.ceil(((2*lmax+1)-np.sqrt((2*lmax+1)*(2*lmax+1)-4*(i-lmax)+1))/2))
        resi[i]=np.sqrt(2)*np.real(nr[i])*(-1)**(mi*mi)
        resi[i+1]=np.sqrt(2)*np.imag(nr[i])*(-1)**(mi*mi)
    return resi

cpdef np.ndarray[np.complex64_t, ndim=1] _buildLm_f(np.ndarray[np.float32_t,
ndim=1] nr, np.int_t lmax):
    cdef np.int size = (len(nr)-lmax-1)/2+lmax+1

    cdef np.ndarray res=np.zeros([size], dtype=np.complex64)
    cdef np.ndarray temp=np.zeros([len(nr)], dtype=np.complex64)
    res[0:lmax+1] = nr[0:lmax+1]

    temp = _inverseRealify_f(nr, lmax)

    for i in xrange(0,len(temp)-lmax-1,2):
        res[i/2+lmax+1] = temp[i+lmax+1]
    return res

cpdef np.ndarray[np.complex64_t, ndim=1] _inverseRealify_f(np.ndarray[np.float32_t, ndim=1] nr, np.int_t lmax):
    cdef np.ndarray resi=np.zeros([len(nr)], dtype=np.complex64)
    resi[0:lmax+1] = np.real(nr[0:lmax+1])

    for i in xrange(lmax+1,len(nr),2):
        mi =  int(np.ceil(((2*lmax+1)-np.sqrt((2*lmax+1)*(2*lmax+1)-4*(i-lmax)+1))/2))
        resi[i]=(-1)**mi/np.sqrt(2)*(nr[i]+1j*nr[i+1])
        resi[i+1]=1/np.sqrt(2)*(nr[i]-1j*nr[i+1])
    return resi


cpdef np.ndarray[np.complex128_t, ndim=1] _buildLm(np.ndarray[np.float64_t,
ndim=1] nr, np.int_t lmax):
    cdef np.int size = (len(nr)-lmax-1)/2+lmax+1

    cdef np.ndarray res=np.zeros([size], dtype=np.complex128)
    cdef np.ndarray temp=np.zeros([len(nr)], dtype=np.complex128)
    res[0:lmax+1] = nr[0:lmax+1]

    temp = _inverseRealify(nr, lmax)

    for i in xrange(0,len(temp)-lmax-1,2):
        res[i/2+lmax+1] = temp[i+lmax+1]
    return res

cpdef np.ndarray[np.complex128_t, ndim=1] _inverseRealify(np.ndarray[np.float64_t, ndim=1] nr, np.int_t lmax):
    cdef np.ndarray resi=np.zeros([len(nr)], dtype=np.complex128)
    resi[0:lmax+1] = np.real(nr[0:lmax+1])

    for i in xrange(lmax+1,len(nr),2):
        mi =  int(np.ceil(((2*lmax+1)-np.sqrt((2*lmax+1)*(2*lmax+1)-4*(i-lmax)+1))/2))
        resi[i]=(-1)**mi/np.sqrt(2)*(nr[i]+1j*nr[i+1])
        resi[i+1]=1/np.sqrt(2)*(nr[i]-1j*nr[i+1])
    return resi

