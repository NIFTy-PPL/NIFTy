import numpy as np
cimport numpy as np

cpdef buildIdx(np.ndarray[np.complex128_t, ndim=1] nr, np.int_t lmax):
    cdef np.int size = (lmax+1)*(lmax+1)

    cdef np.ndarray res=np.zeros([size], dtype=np.complex128)
    cdef np.ndarray final=np.zeros([size], dtype=np.float64)
    res[0:lmax+1] = nr[0:lmax+1]

    for i in xrange(len(nr)-lmax-1):
        res[i*2+lmax+1] = nr[i+lmax+1]
        res[i*2+lmax+2] = np.conjugate(nr[i+lmax+1])
    final = realify(res, lmax)
    return final

cpdef np.ndarray[np.float64_t, ndim=1] realify(np.ndarray[np.complex128_t, ndim=1] nr, np.int_t lmax):
    cdef np.ndarray resi=np.zeros([len(nr)], dtype=np.float64)

    resi[0:lmax+1] = np.real(nr[0:lmax+1])

    for i in xrange(lmax+1,len(nr),2):
        # m calculation print i,(i-lmax)/2+1,(2*lmax+1)*(2*lmax+1)-4*(i-lmax)+1
        mi =  int(np.ceil(((2*lmax+1)-np.sqrt((2*lmax+1)*(2*lmax+1)-4*(i-lmax)+1))/2))
        resi[i]=np.sqrt(2)*np.real(nr[i])*(-1)**(mi*mi)
        resi[i+1]=np.sqrt(2)*np.imag(nr[i])*(-1)**(mi*mi)
    return resi

cpdef buildLm(np.ndarray[np.float64_t, ndim=1] nr, np.int_t lmax):
    cdef np.int size = (len(nr)-lmax-1)/2+lmax+1

    cdef np.ndarray res=np.zeros([size], dtype=np.complex128)
    cdef np.ndarray temp=np.zeros([len(nr)], dtype=np.complex128)
    res[0:lmax+1] = nr[0:lmax+1]

    temp = inverseRealify(nr, lmax)

    for i in xrange(0,len(temp)-lmax-1,2):
        res[i/2+lmax+1] = temp[i+lmax+1]
    return res

cpdef np.ndarray[np.complex128_t, ndim=1] inverseRealify(np.ndarray[np.float64_t, ndim=1] nr, np.int_t lmax):
    cdef np.ndarray resi=np.zeros([len(nr)], dtype=np.complex128)
    resi[0:lmax+1] = np.real(nr[0:lmax+1])

    for i in xrange(lmax+1,len(nr),2):
        # m calculation print i,(i-lmax)/2+1,(2*lmax+1)*(2*lmax+1)-4*(i-lmax)+1
        mi =  int(np.ceil(((2*lmax+1)-np.sqrt((2*lmax+1)*(2*lmax+1)-4*(i-lmax)+1))/2))
        resi[i]=(-1)**mi/np.sqrt(2)*(nr[i]+1j*nr[i+1])
        resi[i+1]=1/np.sqrt(2)*(nr[i]-1j*nr[i+1])
    return resi

