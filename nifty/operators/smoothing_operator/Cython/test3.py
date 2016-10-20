import pyximport; pyximport.install()
import extended

import numpy as np


print "///////////////////////////////////////First thing ////////////////////////"

arr = np.ones(5, dtype=np.float)
arr[2] = 10.0
mk = np.arange(5, dtype=np.float)
mi = 3.0
smooth = 1.0

a = extended.GaussianKernel(arr,mk,mi,smooth)

print a

print "///////////////////////////////////////Second thing ////////////////////////"


n=12

ksq=np.sqrt(np.arange(n), dtype=np.float)
kk=np.arange(n, dtype=np.float)
power = np.ones(n, dtype=np.float)
# power[0][4][4]=1000
# power[1][4][4]=1000
# power[2][4][4]=1000
# power[3][4][4]=1000
power[n/2]=100
# power[5][4][4]=1000
# power[6][4][4]=1000
# power[7][4][4]=1000
k = kk
sigma=k[1]-k[0]
mirrorsize=7
startindex=mirrorsize/2
endindex=n-mirrorsize/2
print power, k, power.shape

smooth = extended.apply_kernel_along_array(power, startindex, endindex, k, sigma)

print smooth

print "///////////////////////////////////////Third thing ////////////////////////"
n=10
ksq=np.sqrt(np.arange(n))
kk=[np.arange(0,1,1.0/n)]*n**2
power = np.ones(n**3).reshape((n,n,n))
# power[0][4][4]=1000
# power[1][4][4]=1000
# power[2][4][4]=1000
# power[3][4][4]=1000
power[n/2][n/2][n/2]=10000
# power[5][4][4]=1000
# power[6][4][4]=1000
# power[7][4][4]=1000
distances = np.asarray(kk,dtype=np.float64)
sigma=k[1]-k[0]
mirrorsize=5
startindex=mirrorsize/2
endindex=n-mirrorsize/2
print k, power, power.shape
# smooth = extended.smooth_something(datablock=power, axis=(2),
#                                    startindex=startindex, endindex=endindex,
#                                    kernelfunction=extended.GaussianKernel, k=k,
#                                    sigma=sigma)


smooth = extended.apply_along_axis(2, power,
                               startindex, endindex, distances,
                               sigma)

print "Smoooooth", smooth