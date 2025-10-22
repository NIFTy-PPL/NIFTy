# NIFTy Introduction

# Introduction
This book provides an introduction to the Bayesian inference library [NIFTy](https://gitlab.mpcdf.mpg.de/ift/nifty).
NIFTy is designed to enable inference in very high-dimensional posterior distributions, as required for image or 3D volume reconstruction applications.
It has already been successfully applied to problems with hundreds of millions of parameters.
To support such applications, NIFTy includes at its core scalable Gaussian process models and fast variational inference algorithms.
There are two variants of NIFTy, called `NIFTy.re` and `NIFTy.cl`.
This introduction focuses on the [JAX](https://github.com/jax-ml/jax)-based variant `NIFTy.re`.
However, many of the concepts introduced here also apply to `NIFTy.cl`.


The content of this book is based on the example notebooks available on the [NIFTy webpage](https://ift.pages.mpcdf.de/nifty/).
In addition, the webpage includes further resources such as the [API reference](https://ift.pages.mpcdf.de/nifty/mod/nifty.html).
You can also find additional introductory material in the [demos folder](https://gitlab.mpcdf.mpg.de/ift/nifty/-/tree/main/demos?ref_type=heads) of the GitLab repository.
The source code of NIFTy and installation instructions are available on the [GitLab page](https://gitlab.mpcdf.mpg.de/ift/nifty).