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

    print "mpower", mpower
    mk = np.r_[(2*k[0]-k[1:nmirror][::-1]),k,(2*k[-1]-k[-nmirror:-1][::-1])]
    mdk = np.r_[0.5*(mk[1]-mk[0]),0.5*(mk[2:]-mk[:-2]),0.5*(mk[-1]-mk[-2])]

    p_smooth = np.empty(mpower.shape)
    for i in xrange(len(p_smooth)):
        l = i-int(2*smooth_length/mdk[i])-1
        l = max(l,0)
        u = i+int(2*smooth_length/mdk[i])+2
        u = min(u,len(p_smooth))
        C = np.exp(-(mk[l:u]-mk[i])**2/(2.*smooth_length**2))*mdk[l:u]
        p_smooth[i] = np.sum(C*mpower[l:u])/np.sum(C)

    p_smooth = p_smooth[nmirror - 1:-nmirror + 1]

#    dk = 0.5*(k[2:] - k[:-2])
#    dk = np.r_[0.5*(k[1]-k[0]),dk]
#    dk = np.r_[dk,0.5*(k[-1]-k[-2])]
#    if (smooth_length is None) or (smooth_length < 0):
#        smooth_length = k[1]-k[0]
#
#    p_smooth = np.empty(power.shape)
#    for i in xrange(len(p_smooth)):
#        l = i-int(2*smooth_length/dk[i])-1
#        l = max(l,0)
#        u = i+int(2*smooth_length/dk[i])+2
#        u = min(u,len(p_smooth))
#        C = np.exp(-(k[l:u]-k[i])**2/(2.*smooth_length**2))*dk[l:u]
#        p_smooth[i] = np.sum(C*power[l:u])/np.sum(C)

    if (exclude > 0):
        p_smooth = np.r_[excluded_power,p_smooth]
    return p_smooth