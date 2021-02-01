import nifty7 as ift
import numpy as np


"""THIS IS A DRAFT VERSION OF A TEST FOR INTERPOLATORS"""

res = 64
vol = 2
sp = ift.RGSpace([res, res], [vol/res, vol/res])
# sp = ift.RGSpace([res, res])

mg = np.mgrid[(slice(0,res),)*2]
mg = np.array(list(map(np.ravel,mg)))

dist = [list(sp.distances)]
dist = np.array(dist).reshape(-1, 1)

sampling_points = dist * mg
R = ift.FFTInterpolator(sp, sampling_points)
linInp = ift.LinearInterpolator(sp, sampling_points)

ift.extra.check_linear_operator(linInp, atol=1e-7, rtol=1e-7)
ift.extra.check_linear_operator(R, atol=1e-7, rtol=1e-7)

inp = ift.from_random(R.domain)
out = R(inp).val
out1 = linInp(inp).val

np.testing.assert_allclose(out, inp.val.reshape(-1))
np.testing.assert_allclose(out1, inp.val.reshape(-1))
np.testing.assert_allclose(out, out1)

sampling_points = np.array([[0.25], [0.]])
R = ift.FFTInterpolator(sp, sampling_points)
R1 = ift.LinearInterpolator(sp, sampling_points)

p = ift.Plot()
p.add(R.adjoint(ift.full(R.target, 1)), title="FFT")
p.add(R1.adjoint(ift.full(R.target, 1)), title="Linear")
p.output(name="debug.png", ny=1, xsize=12)



#TODO Generate one Fourriermode, read out between gridpoints, check if right value
