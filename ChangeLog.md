Changes since NIFTy 6
=====================

New parametric amplitude model
------------------------------

The `ift.CorrelatedFieldMaker` now features two amplitude models. In addition
to the non-parametric one, one may choose to use a Matern kernel instead. The
method is aptly named `add_fluctuations_matern`. The major advantage of the
parametric model is its more intuitive scaling with the size of the position
space.

CorrelatedFieldMaker interface change
-------------------------------------

The interface of `ift.CorrelatedFieldMaker` changed and instances of it may now
be instantiated directly without the previously required `make` method. Upon
initialization, no zero-mode must be specified as the normalization for the
different axes of the power respectively amplitude spectrum now only happens
once in the `finalize` method. There is now a new call named
`set_amplitude_total_offset` to set the zero-mode. The method accepts either an
instance of `ift.Operator` or a tuple parameterizing a log-normal parameter.
Methods which require the zero-mode to be set raise a `NotImplementedError` if
invoked prior to having specified a zero-mode.

Furthermore, the interface of `ift.CorrelatedFieldMaker.add_fluctuations`
changed; it now expects the mean and the standard deviation of their various
parameters not as separate arguments but as a tuple. The same applies to all
new and renamed methods of the `CorrelatedFieldMaker` class.

Furthermore, it is now possible to disable the asperity and the flexibility
together with the asperity in the correlated field model. Note that disabling
only the flexibility is not possible.

Additionally, the parameters `flexibility`, `asperity` and most importantly
`loglogavgslope` refer to the power spectrum instead of the amplitude now.
For existing codes that means that both values in the tuple `loglogavgslope`
and `flexibility` need to be doubled. The transformation of the `asperity`
parameter is nontrivial.

SimpleCorrelatedField
---------------------

A simplified version of the correlated field model was introduced which does not
allow for multiple power spectra, the presence of a degree of freedom parameter
`dofdex`, or `total_N` larger than zero. Except for the above mentioned
limitations, it is equivalent to `ift.CorrelatedFieldMaker`. Hence, if one
wants to understand the implementation idea behind the model, it is easier to
grasp from reading `ift.SimpleCorrelatedField` than from going through
`ift.CorrelatedFieldMaker`.

Change in external dependencies
-------------------------------

Instead of the optional external packages `pypocketfft` and `pyHealpix`, NIFTy
now uses the DUCC package (<https://gitlab.mpcdf.mpg.de/mtr/ducc)>,
which is their successor.


Naming of operator tests
------------------------

The implementation tests for nonlinear operators are now available in
`ift.extra.check_operator()` and for linear operators
`ift.extra.check_linear_operator()`.

MetricGaussianKL interface
--------------------------

`mirror_samples` is not set by default anymore.


GeoMetricKL
-----------

A new posterior approximation scheme, called geometric Variational Inference
(geoVI) was introduced. `GeoMetricKL` extends `MetricGaussianKL` in the sense
that it uses (non-linear) geoVI samples instead of (linear) MGVI samples.
`GeoMetricKL` can be configured such that it reduces to `MetricGaussianKL`.
`GeoMetricKL` is now used in `demos/getting_started_3.py` and a visual
comparison to MGVI can be found in `demos/variational_inference_visualized.py`.
For further details see (<https://arxiv.org/abs/2105.10470>).


LikelihoodEnergyOperator
------------------------

A new subclass of `EnergyOperator` was introduced and all `EnergyOperator`s that
are likelihoods are now `LikelihoodEnergyOperator`s. A
`LikelihoodEnergyOperator` has to implement the function `get_transformation`,
which returns a coordinate transformation in which the Fisher metric of the
likelihood becomes the identity matrix. This is needed for the `GeoMetricKL`
algorithm.


Remove gitversion interface
---------------------------

Since we provide proper nifty releases on PyPI now, the gitversion interface is
not supported any longer.


Changes since NIFTy 5
=====================

Minimum Python version increased to 3.6
---------------------------------------


New operators
-------------

In addition to the below changes, the following operators were introduced:

* UniformOperator: Transforms a Gaussian into a uniform distribution
* VariableCovarianceGaussianEnergy: Energy operator for inferring covariances
* MultiLinearEinsum: Multi-linear version of numpy's einsum with derivates
* LinearEinsum: Linear version of numpy's einsum with one free field
* PartialConjugate: Conjugates parts of a multi-field
* SliceOperator: Geometry preserving mask operator
* SplitOperator: Splits a single field into a multi-field
* MatrixProductOperator: Applies matrices (scipy.sparse, numpy) to fields
* IntegrationOperator: Integrates over subspaces of fields

FFT convention adjusted
-----------------------

When going to harmonic space, NIFTy's FFT operator now uses a minus sign in the
exponent (and, consequently, a plus sign on the adjoint transform). This
convention is consistent with almost all other numerical FFT libraries.

Interface change in EndomorphicOperator.draw_sample()
-----------------------------------------------------

Both complex-valued and real-valued Gaussian probability distributions have
Hermitian and positive endomorphisms as covariance. Just by looking at an
endomorphic operator itself it is not clear whether it is viewed as covariance
for real or complex Gaussians when a sample of the respective distribution shall
be drawn. Therefore, we introduce the method `draw_sample_with_dtype()` which
needs to be given the data type of the probability distribution. This function
is implemented for all operators which actually draw random numbers
(`DiagonalOperator` and `ScalingOperator`). The class `SamplingDtypeSetter` acts
as a wrapper for this kind of operators in order to fix the data type of the
distribution. Samples from these operators can be drawn with `.draw_sample()`.
In order to dive into those subtleties I suggest running the following code and
playing around with the dtypes.

```
import nifty7 as ift
import numpy as np

dom = ift.UnstructuredDomain(5)
dtype = [np.float64, np.complex128][1]
invcov = ift.ScalingOperator(dom, 3)
e = ift.GaussianEnergy(mean=ift.from_random(dom, 'normal', dtype=dtype),
                       inverse_covariance=invcov)
pos = ift.from_random(dom, 'normal', dtype=np.complex128)
lin = e(ift.Linearization.make_var(pos, want_metric=True))
met = lin.metric
print(met)
print(met.draw_sample())
```

New approach for sampling complex numbers
=========================================

When calling draw_sample_with_dtype with a complex dtype,
the variance is now used for the imaginary part and real part separately.
This is done in order to be consistent with the Hamiltonian.
Note that by this,
```
np.std(ift.from_random(domain, 'normal', dtype=np.complex128).val)
````
does not give 1, but sqrt(2) as a result.


MPI parallelisation over samples in MetricGaussianKL
----------------------------------------------------

The classes `MetricGaussianKL` and `MetricGaussianKL_MPI` have been unified
into one `MetricGaussianKL` class which has MPI support built in.

New approach for random number generation
-----------------------------------------

The code now uses `numpy`'s new `SeedSequence` and `Generator` classes for the
production of random numbers (introduced in numpy 1.17. This greatly simplifies
the generation of reproducible random numbers in the presence of MPI parallelism
and leads to cleaner code overall. Please see the documentation of
`nifty7.random` for details.


Interface Change for from_random and OuterProduct
-------------------------------------------------

The sugar.from_random, Field.from_random, MultiField.from_random now take domain
as the first argument and default to 'normal' for the second argument.
Likewise OuterProduct takes domain as the first argument and a field as the second.

Interface Change for non-linear Operators
-----------------------------------------

The method `Operator.apply()` takes a `Linearization` or a `Field` or a
`MultiField` as input. This has not changed. However, now each non-linear
operator assumes that the input `Linearization` comes with an identity operator
as jacobian. Also it is assumed that the `apply()` method returns a
`Linearization` with the jacobian of the operator itself. The user is not in
charge anymore of stacking together the jacobians of operator chains. Something
like `x.jac` should not appear in any self-written `apply()` methods. The method
`Operator._check_input` tests if this condition is met. The same goes for the
metric. There is no need anymore to call `SandwichOperator` in an `apply()`
method when implementing new energies. This change should not lead to unexpected
behaviour since both `Operator._check_input()` and
`extra.check_jacobian_consistency()` tests for the new conditions to be
fulfilled.

Special functions for complete Field reduction operations
---------------------------------------------------------

So far, reduction operations called on Fields (like `vdot`, `sum`, `integrate`,
`mean`, `var`, `std`, `prod` etc.) returned a scalar when the reduction was
carried out over all domains, and otherwise a `Field`.
Having the data type of the returned value depend on input parameters is
extremely confusing, so all of these reduction operations now always return a
Field. We also introduced another set of reduction operations which always
operate over all subdomains and therefore don't take a `spaces` argument; they
are named `s_vdot`, `s_sum` etc. and always return a scalar.

Updates regarding correlated fields
-----------------------------------

The most commonly used model for homogeneous and isotropic correlated fields in
nifty5 has been `SLAmplitude` combined with `CorrelatedField`. This model
exhibits unintuitive couplings between its parameters and as been replaced
by `CorrelatedFieldMaker` in NIFTy 6. This model aims to conceptionally provide
the same functionality. However, internally it works quite differently. Therefore,
specific classes for `SLAmplitude` like `LogRGSpace`, `QHTOperator`, `ExpTransform`,
`SlopeOperator`, `SymmetrizingOperator`, `CepstrumOperator`, `CorrelatedField`
and `MfCorrelatedField` are not needed anymore and have been removed. In general,
`CorrelatedFieldMaker` feels to be better conditioned leading to faster convergence
but it is hard to make explicit tests since the two approaches cannot be mapped
onto each other exactly. We experienced that preconditioning in the `MetricGaussianKL`
via `napprox` breaks the inference scheme with the new model so `napprox` may not
be used here.

Removal of the standard MPI parallelization scheme:
---------------------------------------------------

When several MPI tasks are present, NIFTy5 distributes every Field over these
tasks by splitting it along the first axis. This approach to parallelism is not
very efficient, and it has not been used by anyone for several years, so we
decided to remove it, which led to many simplifications within NIFTy.
User-visible changes:
- the methods `to_global_data`, `from_global_data`, `from_local_data` and
  the property `local_data` have been removed from `Field` and `MultiField`.
  Instead there are now the property `val` (returning a read-only numpy.ndarray
  for `Field` and a dictionary of read-only numpy.ndarrays for `MultiField`) and
  the method `val_rw()` (returning the same structures with writable copies of
  the arrays). Fields and MultiFields can be created from such structures using
  the static method `from_raw`
- the functions `from_global_data` and `from_local_data` in `sugar` have been
  replaced by a single function called `makeField`
- the property `local_shape` has been removed from `Domain` (and subclasses)
  and `DomainTuple`.
