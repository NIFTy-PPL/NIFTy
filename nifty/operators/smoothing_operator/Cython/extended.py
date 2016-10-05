import numpy as np

def smooth_power_2s(power, k, exclude=1, smooth_length=None):

    if smooth_length == 0:
        # No smoothing requested, just return the input array.
        return power

    if (exclude > 0):
        k = k[exclude:]
        excluded_power = np.copy(power[:exclude])
        power = power[exclude:]

    if (smooth_length is None) or (smooth_length < 0):
        smooth_length = k[1]-k[0]

    nmirror = int(5*smooth_length/(k[1]-k[0]))+2
	
    print "nmirror", nmirror

    mpower = np.r_[np.exp(2*np.log(power[0])-np.log(power[1:nmirror][::-1])),power,np.exp(2*np.log(power[-1])-np.log(power[-nmirror:-1][::-1]))]


    mk = np.r_[(2*k[0]-k[1:nmirror][::-1]),k,(2*k[-1]-k[-nmirror:-1][::-1])]
    mdk = np.r_[0.5*(mk[1]-mk[0]),0.5*(mk[2:]-mk[:-2]),0.5*(mk[-1]-mk[-2])]
    print "mpower", mpower
    p_smooth = np.empty(mpower.shape)
    print "p_smooth", p_smooth
    for i in xrange(len(p_smooth)):
        l = i-int(2*smooth_length/mdk[i])-1
        l = max(l,0)
        u = i+int(2*smooth_length/mdk[i])+2
        u = min(u,len(p_smooth))
        print "i", i, "l", l, "u", u
        C = np.exp(-(mk[l:u]-mk[i])**2/(2.*smooth_length**2))*mdk[l:u]
        p_smooth[i] = np.sum(C*mpower[l:u])/np.sum(C)
        # print "p_smooth[",i,"] = ",p_smooth[i]

    print "p_smooth2", " all ", p_smooth
    p_smooth = p_smooth[nmirror - 1:-nmirror + 1]
    print "p_smooth3", p_smooth
    print "p_smooth length ", len(p_smooth)

    if (exclude > 0):
        p_smooth = np.r_[excluded_power,p_smooth]
    return p_smooth

def GaussianKernel(mpower, mk, mu,smooth_length):
    C = np.exp(-(mk - mu) ** 2 / (2. * smooth_length ** 2))
    return np.sum(C * mpower) / np.sum(C)


def smoothie(power, startindex, endindex, k, exclude=1, smooth_length=None):

    if smooth_length == 0:
        # No smoothing requested, just return the input array.
        return power

    excluded_power = []
    if (exclude > 0):
        k = k[exclude:]
        excluded_power = np.copy(power[:exclude])
        power = power[exclude:]

    if (smooth_length is None) or (smooth_length < 0):
        smooth_length = k[1]-k[0]

    p_smooth = np.empty(endindex-startindex)
    for i in xrange(startindex, endindex):
        l = max(i-int(2*smooth_length)-1,0)
        u = min(i+int(2*smooth_length)+2,len(p_smooth))
        p_smooth[i-startindex] = GaussianKernel(power[l:u], k[l:u], k[i],
                                          smooth_length)

    if (exclude > 0):
        p_smooth = np.r_[excluded_power,p_smooth]
    return p_smooth


def smooth_something(datablock, axis=0, startindex=None, endindex=None,
                     kernelfunction=lambda x:x, k=None, sigma=None):
    if startindex == None:
        startindex=0
    if endindex == None:
        endindex=len(datablock)
    print kernelfunction
    return np.apply_along_axis(smoothie, axis, datablock,
                               startindex=startindex, endindex=endindex, k=k,
                               smooth_length=sigma)

    