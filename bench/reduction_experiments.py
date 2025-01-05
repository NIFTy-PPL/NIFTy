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
import jax
import numpy as np
from cupyx.profiler import benchmark, profile

shape = 512, 512, 16

a = np.random.random(shape).astype(np.float32)
a_jax = jnp.array(a)
a_cp = cp.asarray(a)
b = np.random.random(shape).astype(np.float32)
b_jax = jnp.array(b)
b_cp = cp.asarray(b)

assert a_jax.device.id == 0
assert a_cp.device.id == 0


print("##########")
print("# sum(a) #")
print("##########")


@jax.jit
def f_jax(a):
    return jnp.sum(a)


def f_cp(a):
    return cp.sum(a)


f_cp1 = cp.ReductionKernel(
    "T a",
    "T out",
    "a",  # Do nothing
    "a + b",  # Reduction function
    "out = a",
    "0",
    "mysum",
)


# Test that all numpy, jax and cupy compute the same values
ref = np.sum(np.array(a_jax))
res_jax = f_jax(a_jax)
res_cp = f_cp(a_cp)
res_cp1 = f_cp1(a_cp)
np.testing.assert_allclose(ref, np.array(res_jax), rtol=1e-2)
np.testing.assert_allclose(ref, cp.asnumpy(res_cp), rtol=1e-2)
np.testing.assert_allclose(ref, cp.asnumpy(res_cp1), rtol=1e-2)

# Profiling
print(benchmark(f_jax, (a_jax,), name="jax", max_duration=1))
print(benchmark(f_cp, (a_cp,), name="cupy", max_duration=1))
print(benchmark(f_cp1, (a_cp,), name="cupy.ReductionKernel", max_duration=2))


print()
print("############")
print("# sum(a+b) #")
print("############")


@jax.jit
def f_jax(a, b):
    return jnp.sum(a + b)


def f_cp(a, b):
    return cp.sum(a + b)


f_cp1 = cp.ReductionKernel(
    "T a, T b",
    "T out",
    "a + b",
    "a + b",
    "out = a",
    "0",
    "mysum",
)


# Test that all numpy, jax and cupy compute the same values
ref = np.sum(np.array(a_jax) + np.array(b_jax))
res_jax = f_jax(a_jax, b_jax)
res_cp = f_cp(a_cp, b_cp)
res_cp1 = f_cp1(a_cp, b_cp)
np.testing.assert_allclose(ref, np.array(res_jax), rtol=1e-2)
np.testing.assert_allclose(ref, cp.asnumpy(res_cp), rtol=1e-2)
np.testing.assert_allclose(ref, cp.asnumpy(res_cp1), rtol=1e-2)

# Profiling
print(benchmark(f_jax, (a_jax, b_jax), name="jax", max_duration=2))
print(benchmark(f_cp, (a_cp, b_cp), name="cupy", max_duration=2))
print(benchmark(f_cp1, (a_cp, b_cp), name="cupy.ReductionKernel", max_duration=2))
