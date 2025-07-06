Changes since list tagged version
=================================


Changes since NIFTy 8
=====================

General
-------
Minimum Python version increased to 3.10


ift.optimize_kl: fresh_stochasticity
------------------------------------
Introduce the new option `fresh stochasticity` to control whether new randomness
is used at each global iteration. It accepts a boolean (or callable returning
booleans). Setting it to `False` in later iterations can help reduce stochastic
fluctuations and improve convergence behavior.

The optimize_kl config system supports this option as well. Note that old config
files need to be updated: each `optimization` section requires a
`fresh_stochasticity` entry now. To reproduce old behaviour, set
`fresh_stochasticity` to `True` for all iterations.

On a technical level, the RNG state is now saved once at the start (instead of
per iteration), and restored from a unified file (`nifty_random_state`) when
resuming. The per-iteration random state save/load via
`nifty_random_state_<iteration>` has been removed to simplify state management.


ift.estimate_evidence_lower_bound
---------------------------------
Renamed `batch_size` to `n_batches` for clarity. Improved batch logic.


ift.AnyArray: Add general GPU backend numpy-based nifty
-------------------------------------------------------

The old numpy-based nifty is now GPU-compatible. In the best case, you can
specify `device_id=0` for `optimize_kl` and your optimization runs on the GPU.

The basic idea is that `ift.Field.val` is not a numpy array anymore, but rather
a so-called `ift.AnyArray`. It provides an abstraction layer around, e.g., a
`np.ndarray` but it can also hold an ndarray of the GPU (`cupy.ndarray`).

`ift.Field`, `ift.MultiField` and `ift.AnyArray` provide the method
`.at(device_id)` which returns an instance of the respective class on the
specified device. `device_id=-1` corresponds to the host, `device_id=0` is the
first GPU, and so on. If you need the underlying `np.ndarray` of a `Field`,
irrespective of where it is stored, use `.asnumpy()`.

In practice this means that you may need to replace many `.val` calls, e.g., in
plotting routines with `.asnumpy()`. This breaks the interface but is intended
to avoid silent failures.


ift.logger: Set default logging level to INFO
---------------------------------------------

The default debugging level for output to `stdout` has been increased from
`DEBUG` to `INFO`. It can be changed back by the user by running, e.g.,

```
import logging
import nifty as ift

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ift.logger.addHandler(ch)
```


ift.optimize_kl
---------------
Since `optimize_kl` takes so many arguments, we force the user now to pass them
by name except for the first five (`likelihood_enery`, `total_iterations`,
`n_samples`, `kl_minimizer`, `sampling_iteration_controller`).


ift.re: Stabilize ICR at the cost of disallowing using old ICR reconstructions
------------------------------------------------------------------------------
ICR's refinement now uses `eigsh` based matrix inversions and square roots,
significantly stabilizing the scheme. However, previous latent parameters for
ICR will now produce different results, effectively disallowing the use of old
reconstruction results with the new `eigsh` based refinement. To avoid silent
breakage, we renamed `RefinementField` to the more descriptive `ChartedField`.


ift.optimize_kl: indexing
-------------------------
Make the iteration number a counter instead of an index, i.e., initialize at
zero instead of at negative one.


ift & ift.re: Hartley convention
--------------------------------
The Hartley convention can now be configured via `jft.config.update`.



Changes since NIFTy 7
=====================

Rename FinuFFT to Nufft
-----------------------

Add CountingOperator
--------------------
It counts how often an operator is called and distinguishes between applications
on Fields and Linearizations. It also keeps track of how often the Jacobian and
its adjoint is called. This operator does not change the field values.

It is used within `optimize_kl` and its report is written to a file within the
`output_directory`.

Introduce Likelihood energies
-----------------------------
The interface of many energy operators has changed such that they know
themselves how to compute normalized residuals. Therefore, `minisanity` takes
only the LikelihooodEnergyOperator and a `SampleList` as an input now. In order
to be able to distinguish in the output of `minisanity` between different
likelihoods, one can assign a `name` to each LikelihoodEnergyOperator. This
`name` has no effect on the inference and is only used inside `minisanity` for
now.
This change makes it possible to call `minisanity` automatically from within
`optimize_kl`.

Add TransposeOperator
---------------------

InverseGammaOperator interface
-------------------------------------

The InverseGammaOperator now also supports initialization via mode and mean.
Additionally it got some properties like `alpha`, `q`, `mode`, `mean`
(for `alpha`>1), and `var` (for `alpha`>2), that return the respective value.

Extension of Operator interface
-------------------------------

For an operator `op`, `op[key]` represents the operator that first applies `op`,
which results in a `MultiField`, and then extracts the `Field` with key `key`
out of that.

Additionally, `op1.ducktape(op2.target) @ op2` attempts to convert the target of
`op2` into the domain of `op1`. Equivalently, one can write: `op1 @
op2.ducktape_left(op1.domain)`. Both include a potential reshape of the `Field`.
Note that only instances of `DomainTuple` (and not `Domain`s) are allowed for
this use of `ducktape`.

For applying operators to random samples, there is now the convenience function
`op.apply_to_random_sample()`. It draws a random sample on the operator's domain
(of dtype `np.float64`, unless specified otherwise) and applies the operator
to it. If the operator expects input values of other dtypes (complex, integer),
this needs to be indicated with the `dtype` keyword argument. No checks are
performed whether the sampled dtype matches the dtype the operator is designed
for. The function accepts the same keyword arguments as `ift.sugar.from_random`.

Minimum Python version increased to 3.7
---------------------------------------

Optimize KL
-----------

The utility function `optimize_kl` has been added. It provides a blueprint for a
typical NIFTy optimization.

By default, after each global iteration the KL energy value history and reduced
chi-square value history are plotted. If the plots are not wanted or if
matplotlib is not available, set the `optimize_kl` keyword arguments
`plot_energy_history` and `plot_minisanity_history` to `False`.

`{Residual,}SampleList.save_to_fits` implied that it would save the standard deviation
but instead saved the variance. This is now fixed as part of [815453f4](https://gitlab.mpcdf.mpg.de/ift/nifty/-/commit/815453f4335188e7383adf29573a0e4aeeb948ac?merge_request_iid=830) which saves the standard deviation.

Unify MetricGaussianKL and GeoMetricKL
--------------------------------------

The constructor methods `MetricGaussianKL` and `GeoMetricKL` have been unified
into the constructor of `SampledKLEnergy` as both associated variational
inference methods MGVI and geoVI only differ in an additional non-linear
sampling step for geoVI. In NIFTy\_7, the `MetricGaussianKL` and `GeoMetricKL`
methods initialized an instance of the internal `_SampledKLEnergy` class, which
has now been externalized in NIFTy\_8 and can be used to construct a
`SampledKLEnergy` to perform variational inference.

Whether linear (MGVI) or non-linear (geoVI) samples should be used is determined
via the `minimizer_sampling` attribute passed on during initialization. If
`minimizer_sampling` is `None`, the sampling step terminates after the linear
part, and the linear (MGVI) samples are used to construct the `SampledKLEnergy`.
In case `minimizer_sampling` is an `ift.DescentMinimizer`, this minimizer is
used to continue the sampling step with the non-linear part necessary to
generate geoVI samples, which are then used in the `SampledKLEnergy`.

`SampledKLEnergy.samples` now returns a `ResidualSampleList` rather than a
generator of samples. Also the samples in a `ResidualSampleList` have the
position now added to the samples already, as opposed to the samples returned by
the generator used in NIFTy\_7.

SampleListBase, SampleList and ResidualSampleList
-------------------------------------------------

New data structure for a list of fields that represents a collection of samples
from a probability distribution. A `SampleList` is an MPI object capable of
handling a distributed set of samples and allows for global access to those
samples via the `iterator` method. It also implements the basic
functionality to compute sample averages via the `sample_stat` method.

The `ResidualSampleList` is a sub-class of `SampleList` which handles samples
via a shared position and residual deviations thereof internally. This
distinction is a required structure for defining the `SampledKLEnergy`.

Evidence Lower Bound
--------------------
Created new function `estimate_evidence_lower_bound` to calculate an estimate
of the evidence lower bound (ELBO) for a model at hand. This can be used for
model comparison.

Sampling dtypes
---------------

We address the general issue of sampling dtypes of covariance operators.
Instances of `EndomorphicOperator` can implement a `draw_sample()` method that
returns a sample from the zero-centered Gaussian distribution with covariance
being the endomorphic operator. If we consider a `DiagonalOperator` as
covariance operator, it must have real positive values on its diagonal
irrespective whether it is associated with a real or a complex Gaussian
distribution. Therefore, this additional piece of information needs to be passed
to the operator. In NIFTy\_6 and NIFTy\_7, this was done with the help of an
operator wrapper, the `SamplingDtypeSetter`. This rather artificial construct
disappears in NIFTy\_8. Now, the sampling dtype needs to be passed to the
constructor of `ScalingOperator`, `DiagonalOperator` or functions like `makeOp`.
For `MultiDomains` either a dict of dtypes can be passed or a single dtype that
applies to all subdomains.

WienerFilterCurvature interface change
--------------------------------------

`ift.WienerFilterCurvature` does not expect sampling dtypes for the likelihood
and the prior anymore. These have to be set during the construction of the
covariance operators.


Minisanity
----------

`ift.extra.minisanity` does not write its results to `ift.logger.info` anymore,
but rather returns its output as a string. Additonally, terminal colors can be
disabled in order to make the output of `ift.extra.minisanity` more readable
when written to a file.

Jax interface
-------------

The interfaces `ift.JaxOperator` and `ift.JaxLikelihoodEnergyOperator` provide
an interface to jax.

Interface change for nthreads
-----------------------------

The number of threads, which are used for the FFTs and ducc in general, used to
be set via `ift.fft.set_nthreads(n)`. This has been moved to
`ift.set_nthreads(n)`. Similarly, `ift.fft.nthreads()` -> `ift.nthreads()`.



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
now uses the DUCC package (<https://gitlab.mpcdf.mpg.de/mtr/ducc>),
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
import nifty as ift
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
`nifty.random` for details.


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
