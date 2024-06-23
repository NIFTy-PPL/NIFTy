# NIFTy - Numerical Information Field Theory

[![pipeline status](https://gitlab.mpcdf.mpg.de/ift/nifty/badges/NIFTy_8/pipeline.svg)](https://gitlab.mpcdf.mpg.de/ift/nifty/-/commits/NIFTy_8)
[![coverage report](https://gitlab.mpcdf.mpg.de/ift/nifty/badges/NIFTy_8/coverage.svg)](https://gitlab.mpcdf.mpg.de/ift/nifty/-/commits/NIFTy_8)

**NIFTy** project homepage: [ift.pages.mpcdf.de/nifty](https://ift.pages.mpcdf.de/nifty/)
 | Found a bug? [github.com/nifty-ppl/nifty/issues](https://github.com/nifty-ppl/nifty/issues)
 | Need help? [github.com/nifty-ppl/nifty/discussions](https://github.com/NIFTy-PPL/NIFTy/discussions)

## Summary

### Description

**NIFTy**, "**N**umerical **I**nformation **F**ield **T**heor<strong>y</strong>", is a Bayesian imaging library.
It is designed to infer the million to billion dimensional posterior distribution in the image space from noisy input data.
At the core of NIFTy lies a set of powerful Gaussian Process (GP) models and accurate Variational Inference (VI) algorithms.

#### Gaussian Processes

One standard tool from the NIFTy toolbox are the structured GP models.
These models usually rely on the harmonic domain being easily accessible, e.g. for pixels spaced on a regular Cartesian grid, the natural choice to represent a stationary kernel is the Fourier domain.
An example, initializing a non-parameteric GP prior for a $128 \times 128$ space with unit volume is shown in the following.

```python
from nifty8 import re as jft

dims = (128, 128)
cfm = jft.CorrelatedFieldMaker("cf")
cfm.set_amplitude_total_offset(offset_mean=2, offset_std=(1e-1, 3e-2))
cfm.add_fluctuations(  # Axis over which the kernle is defined
  dims,
  distances=tuple(1.0 / d for d in dims),
  fluctuations=(1.0, 5e-1),
  loglogavgslope=(-3.0, 2e-1),
  flexibility=(1e0, 2e-1),
  asperity=(5e-1, 5e-2),
  prefix="ax1",
  non_parametric_kind="power",
)
correlated_field = cfm.finalize()  # forward model for a GP prior
```

Not all problems are well described by regularly spaced pixels.
For more complicated pixel spacings, NIFTy features Iterative Charted Refinement, a GP model for arbitrarily deformed spaces.
This model exploits nearest neighbor relations on various coarsening of the discretized modeled space and runs very efficiently on GPUs.
For one dimensional problems with arbitrarily spaced pixel, NIFTy also implements multiple flavors of Gauss-Markov processes.

#### Building Up Complex Models

Models are rarely just a GP prior.
Commonly, a model contains at least several non-linearities that transform the GP prior or combine it with other random variables.
For building more complex models, NIFTy provides a `Model` class that offers a somewhat familiar object-oriented design yet is fully JAX compatible and functional under the hood.
The following code showcases such a model that builds up a slightly more involved model using the objects from the previous example.

```python
from jax import numpy as jnp


class Forward(jft.Model):
  def __init__(self, correlated_field):
    self._cf = correlated_field
    # Track a method with which a random input for the model. This is not
    # strictly required but is usually handy when building deep models.
    super().__init__(init=correlated_field.init)

  def __call__(self, x):
    # NOTE, any kind of masking of the output, non-linear and linear
    # transformation could be carried out here. Models can also combined and
    # nested in any way and form.
    return jnp.exp(self._cf(x))


forward = Forward(correlated_field)

data = jnp.load("data.npy")
lh = jft.Poissonian(data).amend(forward)
```

All GP models in NIFTy as well as all likelihoods are models and their attributes are exposed to JAX, meaning JAX understands what it means if a computation involves `self` or other models.
In other words, `correlated_field`, `forward`, and `lh` from the code snippets shown here are all so-called pytrees in JAX and, e.g., the following is valid code `jax.jit(lambda l, x: l(x))(lh, x0)` with `x0` some arbitrarily chosen valid input to `lh`.
Inspired by [equinox](https://github.com/patrick-kidger/equinox), individual attributes of the class can be marked as non-static or static via `dataclass.field(metadata=dict(static=...))` for the purpose of compiling.
Depending on the value, JAX will either treat the attribute as unknown placeholder or as known concrete attribute and potentially inline it during compiles.
This mechanism is extensively used in likelihoods to avoid inlining large constants such as the data and avoiding expensive re-compiles whenever possible.

#### Variational Inference

NIFTy is built for models with millions to billions of degrees of freedom.
To probe the posterior efficiently and accurately, NIFTy relies on VI.
At the core of the VI methods lie an alternating procedure in which we switch between optimizing the Kullback–Leibler divergence for a specific shape of the variational posterior and updating the shape of the variational posterior.

A typical minimization with NIFTy is shown in the following.
It retrieves six independent, antithetically mirrored samples from the approximate posterior via 25 iterations of alternating between optimization and sample adaption.
The final result is stored in the `samples` variable.
A convenient one-shot wrapper for the below is `jft.optimize_kl`.
By virtue of all modeling tools in NIFTy being written in JAX, it is also possible to combine NIFTy tools with [blackjax](https://blackjax-devs.github.io/blackjax/) or any other posterior sampler in the JAX ecosystem.

```python
from jax import random

key = random.PRNGKey(42)
key, sk = random.split(key, 2)
# NIFTy is agnostic w.r.t. the type of input it gets as long as it supports core
# arithmetic properties. Tell NIFTy to treat our parameter dictionary as a
# vector.
samples = jft.Samples(pos=jft.Vector(lh.init(sk)), samples=None, keys=None)

delta = 1e-4
absdelta = delta * jft.size(samples.pos)

opt_vi = jft.OptimizeVI(lh, n_total_iterations=25)
opt_vi_st = opt_vi.init_state(
  key,
  # Typically on the order of 2-12
  n_samples=lambda i: 1 if i < 2 else (2 if i < 4 else 6),
  # Arguments for the conjugate gradient method used to drawing samples from
  # an implicit covariance matrix
  draw_linear_kwargs=dict(
    cg_name="SL", cg_kwargs=dict(absdelta=absdelta / 10.0, maxiter=100)
  ),
  # Arguements for the minimizer in the nonlinear updating of the samples
  nonlinearly_update_kwargs=dict(
    minimize_kwargs=dict(
      name="SN", xtol=delta, cg_kwargs=dict(name=None), maxiter=5
    )
  ),
  # Arguments for the minimizer of the KL-divergence cost potential
  kl_kwargs=dict(minimize_kwargs=dict(name="M", xtol=delta, maxiter=35)),
  sample_mode=lambda i: "nonlinear_resample" if i < 3 else "nonlinear_update",
)
for i in range(opt_vi.n_total_iterations):
  print(f"Iteration {i+1:04d}")
  # Continuously updates the samples of the approximate posterior distribution
  samples, opt_vi_st = opt_vi.update(samples, opt_vi_st)
  print(opt_vi.get_status_message(samples, opt_vi_st))
```

## Installation

If you only want to use NIFTy in your projects, but not change its source code, the easiest way to install NIFTy is via pip:

```
pip install --user 'nifty8[re]'
```

The line above installs the optional JAX backend termed NIFTy.re in addition to the numpy-based NIFTy.

If you might want to adapt the NIFTy source code, we suggest installing NIFTy as editable python package using the following commands:

```
git clone -b NIFTy_8 https://gitlab.mpcdf.mpg.de/ift/nifty.git
cd nifty
pip install --user --editable '.[re]'
```

## First Steps

For a quick start, you can browse through the [informal introduction](https://ift.pages.mpcdf.de/nifty/user/) or dive into NIFTy by running one of the demonstrations, e.g.:

```
python demos/0_intro.py
```

## Contributing

Contributions are very welcome!
Feel free to reach out early on in the development process e.g. by opening a draft PR or filing an issue, we are happy to help in the development and provide feedback along the way.
Please open an issue first if you think your PR changes current code substantially.
Please format your code according to the existing style used in the file or with black for new files.
To advertise your changes, please update the public documentation and the ChangeLog if your PR affects the public API.
Please add appropriate tests to your PR.

### Building the Documentation

NIFTy's documentation is generated via [Sphinx](https://www.sphinx-doc.org/en/stable/) and is available online at [ift.pages.mpcdf.de/nifty](https://ift.pages.mpcdf.de/nifty/).

To build the documentation locally, run:

```
sudo apt-get install dvipng jupyter-nbconvert texlive-latex-base texlive-latex-extra
pip install --user sphinx jupytext pydata-sphinx-theme myst-parser
cd <nifty_directory>
bash docs/generate.sh
```

To view the documentation, open `docs/build/index.html` in your browser.

Note: Make sure that you reinstall nifty after each change since sphinx imports nifty from the Python path.

### Run the tests

To run the tests, install all optional requirements `'nifty8[all]'` and afterwards run pytest (and create a coverage report) via

```
pytest --cov=nifty8 test
```

If you are writing your own tests, it is often sufficient to just install the optional test dependencies `'nifty8[test]'`. However, to run the full test suit including tests of optional functionality, it is assumed that all optional dependencies are installed.

## Licensing terms

Most of NIFTy is licensed under the terms of the
[GPLv3](https://www.gnu.org/licenses/gpl.html) license with NIFTy.re being a notable exception.
NIFTy.re is licensed under GPL-2.0+ OR BSD-2-Clause.
All of NIFTy is distributed *without any warranty*.

## Citing NIFTy

To cite the probabilistic programming framework NIFTy, please use the citation provided below.
In addition to citing NIFTy itself, please consider crediting the Gaussian process models you used and the inference machinery.
See [the corresponding entry on citing NIFTy in the documentation](https://ift.pages.mpcdf.de/nifty/user/citations.html) for further details.

```
@article{niftyre,
  title     = {Re-Envisioning Numerical Information Field Theory (NIFTy.re): A Library for Gaussian Processes and Variational Inference},
  author    = {Gordian Edenhofer and Philipp Frank and Jakob Roth and Reimar H. Leike and Massin Guerdi and Lukas I. Scheel-Platz and Matteo Guardiani and Vincent Eberle and Margret Westerkamp and Torsten A. Enßlin},
  year      = {2024},
  journal   = {Journal of Open Source Software},
  publisher = {The Open Journal},
  volume    = {9},
  number    = {98},
  pages     = {6593},
  doi       = {10.21105/joss.06593},
  url       = {https://doi.org/10.21105/joss.06593},
}
```
