# NIFTy - Numerical Information Field Theory

[![pipeline status](https://gitlab.mpcdf.mpg.de/ift/nifty/badges/NIFTy_8/pipeline.svg)](https://gitlab.mpcdf.mpg.de/ift/nifty/-/commits/NIFTy_8)
[![coverage report](https://gitlab.mpcdf.mpg.de/ift/nifty/badges/NIFTy_8/coverage.svg)](https://gitlab.mpcdf.mpg.de/ift/nifty/-/commits/NIFTy_8)

**NIFTy** project homepage: [ift.pages.mpcdf.de/nifty](https://ift.pages.mpcdf.de/nifty/)
 | Found a bug? [github.com/nifty-ppl/nifty/issues](https://github.com/nifty-ppl/nifty/issues)
 | Need help? [github.com/nifty-ppl/nifty/discussions](https://github.com/NIFTy-PPL/NIFTy/discussions)

**NIFTy**, "**N**umerical **I**nformation **F**ield **T**heor<strong>y</strong>",
is a Bayesian inference library.  It is designed to compute statistical
properties of high-dimensional posterior probability distributions (tested up to
billions of parameters) from noisy input data.  At the core of NIFTy lies a set
of Gaussian Process (GP) models and Variational Inference (VI) algorithms - in
particular Metric Gaussian Variational Inference (MGVI) and Geometric Gaussian
Variational Inference (geoVI).

There are two independent implementation variants of NIFTy:

- [Re-envisioned NIFTy](#niftyre) (`nifty.re`)
- [Classical NIFTy](#niftycl) (`nifty.cl`)

These variants share lots of functionality:

- Similar VI algorithms
- Similar GP models
- Similar interfaces (e.g., `nifty.cl/re.optimize_kl` and
  `nifty.cl/re.CorrelatedFieldMaker`)
- Both can run on CPUs and GPUs

The major differences between them are:

- Philosophy: `nifty.cl` provides hackable transparent building blocks to
  explore discretization-independent Bayesian inference algorithms with minimal
  dependencies. On the other hand, `nifty.re` is built around JAX, provides a
  more direct numpy-like interface and aims for high performance out of the box.
- Backend: numpy/cupy (`nifty.cl`) vs JAX (`nifty.re`).
- Performance: `nifty.re` leverages JIT from JAX and thereby runs generally
  faster than `nifty.cl`.
- Functionality: `nifty.re` supports HMC and Multi Grids. `nifty.cl` does not
  (yet).
- API: In `nifty.cl` algorithms are implemented independent of the chosen
  discretization scheme represented explicitly by `nifty.cl.Domain`s. `nifty.re`
  provides more direct access to arrays.
- License: `nifty.cl` is distributed under GPL-3.0+. `nifty.re` is distributed
  under BSD-2-Clause or GPL-2.0+.


## Installation
NIFTy releases are distributed on PyPI. The latest version of NIFTy can be installed from source. Depending on the variant `nifty.cl` or `nifty.re` and possible GPU support, the dependencies and exact installation procedure differ as detailed below

### NIFTy.cl

For a minimal installation of `nifty.cl` run:
```
pip install --user 'nifty'
```

The latest version of `nifty.cl` can be installed from the sources:
```
pip install git+https://gitlab.mpcdf.mpg.de/ift/nifty@nifty
```

Both will install the basic required dependencies (numpy and scipy). Often users
may choose to install optional dependencies to enable additional features.

- `ducc0`: Use FFTs directly from ducc and enable several operators implemented
  directly in C++ for speed.
- `cupy`: Enable GPU backend.
- `pyvkfft`: Use vkFFT instead of cufft.
- `mpi4py`: Parallelize computations via MPI.
- `astropy`: Save certain outputs as FITS files.
- `h5py`: Save certain outputs as HDF5 files.
- `matplotlib`: Enable plotting, e.g., via `nifty.cl.Plot`.


### NIFTy.re
For a CPU only installation of `nifty.re` run:
```
pip install --user 'nifty[re]'
```

To install `nifty.re` with GPU support please manually install JAX following the instructions in the [JAX installation guid](https://docs.jax.dev/en/latest/installation.html).

If you might want to adapt the NIFTy source code, we suggest installing NIFTy as editable python package with a command such as:

```
git clone -b nifty https://gitlab.mpcdf.mpg.de/ift/nifty.git
cd nifty
pip install --user --editable '.[re]'
```

## First Steps

For a quick start, you can browse through the [informal
introduction](https://ift.pages.mpcdf.de/nifty/user/code.html) or dive into
NIFTy by running one of the demonstrations, e.g. for `nfity.cl`:
```
python demos/cl/getting_started_1.py
```
or for `nifty.re`:
```
python demos/re/0_intro.py
```



## Run the tests

To run the tests, install all optional requirements `'nifty[all]'` and afterwards run pytest (and create a coverage report) via

```
pytest -n auto --cov=nifty test
```

If you are writing your own tests, it is often sufficient to just install the optional test dependencies `'nifty[test]'`. However, to run the full test suit including tests of optional functionality, it is assumed that all optional dependencies are installed.



## Building the Documentation

NIFTy's documentation is generated via [Sphinx](https://www.sphinx-doc.org/en/stable/) and is available online at [ift.pages.mpcdf.de/nifty](https://ift.pages.mpcdf.de/nifty/).

To build the documentation locally, run:

```
sudo apt-get install dvipng jupyter-nbconvert texlive-latex-base texlive-latex-extra
pip install --user sphinx==8.1.3 jupytext pydata-sphinx-theme myst-parser sphinxcontrib-bibtex
cd <nifty_directory>
bash docs/generate.sh
```

To view the documentation, open `docs/build/index.html` in your browser.

Note: Make sure that you reinstall nifty after each change since sphinx imports nifty from the Python path.


## Contributing

Contributions are very welcome!
Feel free to reach out early on in the development process e.g. by opening a draft PR or filing an issue, we are happy to help in the development and provide feedback along the way.
Please open an issue first if you think your PR changes current code substantially.
Please format your code according to the existing style used in the file or with black for new files.
To advertise your changes, please update the public documentation and the ChangeLog if your PR affects the public API.
Please add appropriate tests to your PR.


## Citing

To cite the probabilistic programming framework NIFTy, please use the citation provided below for `nfity.re` and `nifty.cl`.
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

```
@article{niftycl,
  author        = {{Arras}, Philipp and {Baltac}, Mihai and {Ensslin}, Torsten A. and {Frank}, Philipp and {Hutschenreuter}, Sebastian and {Knollmueller}, Jakob and {Leike}, Reimar and {Newrzella}, Max-Niklas and {Platz}, Lukas and {Reinecke}, Martin and {Stadler}, Julia},
  title         = {{NIFTy5: Numerical Information Field Theory v5}},
  keywords      = {Software},
  howpublished  = {Astrophysics Source Code Library, record ascl:1903.008},
  year          = 2019,
  month         = 03,
  eid           = {ascl:1903.008},
  pages         = {ascl:1903.008},
  archiveprefix = {ascl},
  eprint        = {1903.008}
}
```
