import pyximport; pyximport.install()
import extended

import numpy as np


print "///////////////////////////////////////First thing ////////////////////////"

ksq=np.sqrt(np.arange(8))
k=np.arange(8)
power = np.ones(512).reshape((8,8,8))
power[0][4][4]=1000
power[1][4][4]=1000
power[2][4][4]=1000
power[3][4][4]=1000
power[4][4][4]=1000
power[5][4][4]=1000
power[6][4][4]=1000
power[7][4][4]=1000
print power, k, power.shape
smooth = extended.smooth_something(datablock=power, axis=(2), startindex=None, endindex=None, kernelfunction=extended.smoothie, k=ksq)

print "Smoooooth", smooth

print "///////////////////////////////////////Final thing ////////////////////////"
print "smooth.len == power.len" , smooth.shape, power.shape, power.shape==smooth.shape