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

### Code Variants
There are two implementations of the NIFTy inference paradigms: NIFTy.cl, also known as classical NIFTy, and NIFTy.re.
Both are centered around the same VI algorithms, but NIFTy.re features more advanced models for Gaussian processes.
NIFTy.re is based on the JAX library, which provides automatic differentiation, just-in-time compilation, and model execution on accelerators.
In contrast, NIFTy.cl comes with a self-built automatic differentiation engine that relies on NumPy.
GPU support can be added to NIFTy.cl via the Cupy package.
Both implementations are included in this Python package.
The NIFTy.cl code is in the cl folders, and the NIFTy.re code is in the re folders.


## Installation

If you only want to use NIFTy in your projects, but not change its source code, the easiest way to install NIFTy is via pip. For a minimal installation of NIFTy.cl please execute:

```
pip install --user 'nifty'
```

To install NIFTy.cl with GPU support use the following command:

```
pip install --user 'nifty[cl_gpu]'
```

For the NIFTy.re installation please run:

```
pip install --user 'nifty[re]'
```

To install NIFTy.re with GPU support please manually install JAX following the instructions in the [JAX installation guid](https://docs.jax.dev/en/latest/installation.html).

If you might want to adapt the NIFTy source code, we suggest installing NIFTy as editable python package with a command such as:

```
git clone -b NIFTy_8 https://gitlab.mpcdf.mpg.de/ift/nifty.git
cd nifty
pip install --user --editable '.[re]'
```

## First Steps

For a quick start, you can browse through the [informal introduction](https://ift.pages.mpcdf.de/nifty/user/) or dive into NIFTy by running the scrips in the demos folder.
The subfolders cl and re contain the scripts relevant for the respective NIFTy flavor.


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
pip install --user sphinx==8.1.3 jupytext pydata-sphinx-theme myst-parser sphinxcontrib-bibtex
cd <nifty_directory>
bash docs/generate.sh
```

To view the documentation, open `docs/build/index.html` in your browser.

Note: Make sure that you reinstall nifty after each change since sphinx imports nifty from the Python path.

### Run the tests

To run the tests, install all optional requirements `'nifty8[all]'` and afterwards run pytest (and create a coverage report) via

```
pytest -n auto --cov=nifty8 test
```

If you are writing your own tests, it is often sufficient to just install the optional test dependencies `'nifty8[test]'`. However, to run the full test suit including tests of optional functionality, it is assumed that all optional dependencies are installed.

## Licensing terms

Most of NIFTy is licensed under the terms of the
[GPLv3](https://www.gnu.org/licenses/gpl.html) license with NIFTy.re being a notable exception.
NIFTy.re is licensed under GPL-2.0+ OR BSD-2-Clause.
All of NIFTy is distributed *without any warranty*.

## Citing NIFTy

To cite the probabilistic programming framework NIFTy, please use the citations provided below for the NIFTY.cl and NIFTy.re variants.
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