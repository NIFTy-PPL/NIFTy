import numpy as np
cimport numpy as np

cpdef buildIdx(np.ndarray[np.complex128_t, ndim=1] nr, np.ndarray[np.int_t] l, np.ndarray[np.int_t] m, np.int_t lmax):
    cdef np.int size = (lmax+1)*(lmax+1)

    cdef np.ndarray res=np.zeros([size], dtype=np.complex128)
    cdef np.ndarray final=np.zeros([size], dtype=np.float64)
    res[0:lmax+1] = nr[0:lmax+1]

    cdef np.ndarray resL=np.zeros([size], dtype=np.int)
    resL[0:lmax+1] = np.arange(lmax+1)

    cdef np.ndarray resM=np.zeros([size], dtype=np.int)

    for i in xrange(len(nr)-lmax-1):
        res[i*2+lmax+1] = nr[i+lmax+1]
        res[i*2+lmax+2] = np.conjugate(nr[i+lmax+1])
        resL[i*2+lmax+1] = l[i+lmax+1]
        resL[i*2+lmax+2] = l[i+lmax+1]
        resM[i*2+lmax+1] = m[i+lmax+1]
        resM[i*2+lmax+2] = -m[i+lmax+1]
    final = realify(res,resL, resM)
    return final, resL, resM

cpdef buildLm(np.ndarray[np.float64_t, ndim=1] nr, np.ndarray[np.int_t] l, np.ndarray[np.int_t] m, np.int_t lmax):
    cdef np.int size = (len(nr)-lmax-1)/2+lmax+1

    cdef np.ndarray res=np.zeros([size], dtype=np.complex128)
    cdef np.ndarray temp=np.zeros([len(nr)], dtype=np.complex128)
    res[0:lmax+1] = nr[0:lmax+1]

    cdef np.ndarray resL=np.zeros([size], dtype=np.int)
    resL[0:lmax+1] = np.arange(lmax+1)

    cdef np.ndarray resM=np.zeros([size], dtype=np.int)

    temp = inverseRealify(nr, l, m)

    for i in xrange(0,len(temp)-lmax-1,2):
        res[i/2+lmax+1] = temp[i+lmax+1]
        resL[i/2+lmax+1] = l[i+lmax+1]
        resM[i/2+lmax+1] = m[i+lmax+1]
    return res,resL,resM

cpdef np.ndarray[np.float64_t, ndim=1] realify(np.ndarray[np.complex128_t, ndim=1] nr, np.ndarray[np.int_t] l, np.ndarray[np.int_t] m):
    cdef np.ndarray resi=np.zeros([len(nr)], dtype=np.float64)

    for i in xrange(len(nr)):
        if m[i]<0:
            resi[i]=np.sqrt(2)*np.imag(nr[i-1])*(-1)**(m[i]*m[i])
        elif m[i]>0:
            resi[i]=np.sqrt(2)*np.real(nr[i])*(-1)**(m[i]*m[i])
        else:
            resi[i]=np.real(nr[i])
    return resi


cpdef np.ndarray[np.complex128_t, ndim=1] inverseRealify(np.ndarray[np.float64_t, ndim=1] nr, np.ndarray[np.int_t] l, np.ndarray[np.int_t] m):
    cdef np.ndarray resi=np.zeros([len(nr)], dtype=np.complex128)

    for i in xrange(len(nr)):
        if m[i]<0:
            resi[i]=1/np.sqrt(2)*(nr[i-1]-1j*nr[i])
        elif m[i]>0:
            resi[i]=(-1)**m[i]/np.sqrt(2)*(nr[i]+1j*nr[i+1])
        else:
            resi[i]=np.real(nr[i])
    return resi

