# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2025 Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import cupy as cp
import jax.numpy as jnp
import numpy as np
from cupyx.profiler import benchmark, profile

shape = 512, 512, 16

a = (np.random.random(shape) + 1j*np.random.random(shape)).astype(np.complex64)
a_jax = jnp.array(a)
a_cp = cp.asarray(a)

assert a_jax.device.id == 0
assert a_cp.device.id == 0

def fft_jax(a):
    return jnp.fft.fftn(a, axes=(0, 1)).block_until_ready()

def fft_cp(a):
    return cp.fft.fftn(a, axes=(0, 1))

def fft_vkfft(a):
    from pyvkfft.fft import fftn
    return fftn(a, axes=(0, 1))


# Test that all numpy, jax and cupy compute the same fft
ref = np.fft.fftn(a, axes=(0, 1))
res_jax = fft_jax(a_jax)
res_cp = fft_cp(a_cp)
res_vkfft = fft_vkfft(a_cp)
np.testing.assert_allclose(ref, np.array(res_jax), rtol=1e-2)
np.testing.assert_allclose(ref, cp.asnumpy(res_cp), rtol=1e-2)

# Profiling
print(benchmark(fft_jax, (a_jax,), name="jax", max_duration=2))
print(benchmark(fft_cp, (a_cp,), name="cupy", max_duration=2))
print(benchmark(fft_vkfft, (a_cp,), name="vkfft", max_duration=2))

with profile():
    fft_vkfft(a_cp)
