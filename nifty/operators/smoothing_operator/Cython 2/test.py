import pyximport; pyximport.install()
import extended

import numpy as np


print "///////////////////////////////////////First thing ////////////////////////"

k=np.sqrt(np.arange(600))
power = np.ones(600)
power[300]=1000
print power, k
smooth = extended.smooth_power_2s(power, k)

print "Smoooooth", smooth

print "///////////////////////////////////////Final thing ////////////////////////"
print "smooth.len == power.len" , len(smooth), len(power), len(power)==len(smooth)