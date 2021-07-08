Information Field Theory
========================

Theoretical Background
----------------------

`Information Field Theory <https://www.mpa-garching.mpg.de/ift/>`_ [1]_  (IFT) is information theory, the logic of reasoning under uncertainty, applied to fields.
A field can be any quantity defined over some space, e.g. the air temperature over Europe, the magnetic field strength in the Milky Way, or the matter density in the Universe.
IFT describes how data and knowledge can be used to infer field properties.
Mathematically it is a statistical field theory and exploits many of the tools developed for such.
Practically, it is a framework for signal processing and image reconstruction.

IFT is fully Bayesian.
How else could infinitely many field degrees of freedom be constrained by finite data?

There is a full toolbox of methods that can be used, like the classical approximation (= Maximum a posteriori = MAP), effective action (= Variational Bayes = VI), Feynman diagrams, renormalization, and more.
IFT reproduces many known well working algorithms, which is reassuring.
Also, there were certainly previous works in a similar spirit.
Anyhow, in many cases IFT provides novel rigorous ways to extract information from data.
NIFTy comes with reimplemented MAP and VI estimators.

.. tip:: A didactical introduction to *information field theory* can be found in [2]_. Other resources include the `Wikipedia entry <https://en.wikipedia.org/wiki/Information_field_theory>`_, [3]_ and [4]_.

.. [1] T.A. Enßlin et al. (2009), "Information field theory for cosmological perturbation reconstruction and nonlinear signal analysis", PhysRevD.80.105005, 09/2009; `[arXiv:0806.3474] <https://www.arxiv.org/abs/0806.3474>`_

.. [2] T.A. Enßlin (2019), "Information theory for fields", Annalen der Physik; `[DOI] <https://doi.org/10.1002/andp.201800127>`_, `[arXiv:1804.03350] <https://arxiv.org/abs/1804.03350>`_

.. [3] T.A. Enßlin (2013), "Information field theory", proceedings of MaxEnt 2012 -- the 32nd International Workshop on Bayesian Inference and Maximum Entropy Methods in Science and Engineering; AIP Conference Proceedings, Volume 1553, Issue 1, p.184; `[arXiv:1301.2556] <https://arxiv.org/abs/1301.2556>`_

.. [4] T.A. Enßlin (2014), "Astrophysical data analysis with information field theory", AIP Conference Proceedings, Volume 1636, Issue 1, p.49; `[arXiv:1405.7701] <https://arxiv.org/abs/1405.7701>`_





Free Theory & Implicit Operators
--------------------------------

A free IFT appears when the signal field :math:`{s}` and the noise :math:`{n}` of the data :math:`{d}` are independent, zero-centered Gaussian processes of known covariances :math:`{S}` and :math:`{N}`, respectively,

.. math::

    \mathcal{P}(s,n) = \mathcal{G}(s,S)\,\mathcal{G}(n,N),

and the measurement equation is linear in both signal and noise,

.. math::

    d= R\, s + n,

with :math:`{R}` being the measurement response, which maps the continuous signal field into the discrete data space.

This is called a free theory, as the information Hamiltonian

.. math::

    \mathcal{H}(d,s)= -\log \mathcal{P}(d,s)= \frac{1}{2} s^\dagger S^{-1} s + \frac{1}{2} (d-R\,s)^\dagger N^{-1} (d-R\,s) + \mathrm{const}

is only of quadratic order in :math:`{s}`, which leads to a linear relation between the data and the posterior mean field.

In this case, the posterior is

.. math::

    \mathcal{P}(s|d) = \mathcal{G}(s-m,D)

with

.. math::

    m = D\, j

the posterior mean field,

.. math::

    D = \left( S^{-1} + R^\dagger N^{-1} R\right)^{-1}

the posterior covariance operator, and

.. math::

    j = R^\dagger N^{-1} d

the information source.
The operation in :math:`{m = D\,R^\dagger N^{-1} d}` is also called the generalized Wiener filter.

NIFTy permits to define the involved operators :math:`{R}`, :math:`{R^\dagger}`, :math:`{S}`, and :math:`{N}` implicitly, as routines that can be applied to vectors, but which do not require the explicit storage of the matrix elements of the operators.

Some of these operators are diagonal in harmonic (Fourier) basis, and therefore only require the specification of a (power) spectrum and :math:`{S= F\,\widehat{P_s} F^\dagger}`.
Here :math:`{F = \mathrm{HarmonicTransformOperator}}`, :math:`{\widehat{P_s} = \mathrm{DiagonalOperator}(P_s)}`, and :math:`{P_s(k)}` is the power spectrum of the process that generated :math:`{s}` as a function of the (absolute value of the) harmonic (Fourier) space coordinate :math:`{k}`.
For those, NIFTy can easily also provide inverse operators, as :math:`{S^{-1}= F\,\widehat{\frac{1}{P_s}} F^\dagger}` in case :math:`{F}` is unitary, :math:`{F^\dagger=F^{-1}}`.

These implicit operators can be combined into new operators, e.g. to :math:`{D^{-1} = S^{-1} + R^\dagger N^{-1} R}`, as well as their inverses, e.g. :math:`{D = \left( D^{-1} \right)^{-1}}`.
The invocation of an inverse operator applied to a vector might trigger the execution of a numerical linear algebra solver.

Thus, when NIFTy calculates :math:`{m = D\, j}`, it actually solves :math:`{D^{-1} m = j}` for :math:`{m}` behind the scenes.
The advantage of implicit operators compared to explicit matrices is the reduced memory consumption;
for the reconstruction of just a Megapixel image the latter would already require several Terabytes.
Larger images could not be dealt with due to the quadratic memory requirements of explicit operator representations.

The demo codes `demos/getting_started_1.py` and `demos/Wiener_Filter.ipynb` illustrate this.


Generative Models
-----------------

For more sophisticated measurement situations (involving non-linear measurements, unknown covariances, calibration constants and the like) it is recommended to formulate those as generative models for which NIFTy provides powerful inference algorithms.

In a generative model, all known or unknown quantities are described as the results of generative processes, which start with simple probability distributions, like the uniform, the i.i.d. Gaussian, or the delta distribution.

Let us rewrite the above free theory as a generative model:

.. math::

    s = A\,\xi

with :math:`{A}` the amplitude operator such that it generates signal field realizations with the correct covariance :math:`{S=A\,A^\dagger}` when being applied to a white Gaussian field :math:`{\xi}` with :math:`{\mathcal{P}(\xi)= \mathcal{G}(\xi, 1)}`.

The joint information Hamiltonian for the standardized signal field :math:`{\xi}` reads:

.. math::

    \mathcal{H}(d,\xi)= -\log \mathcal{P}(d,s)= \frac{1}{2} \xi^\dagger \xi + \frac{1}{2} (d-R\,A\,\xi)^\dagger N^{-1} (d-R\,A\,\xi) + \mathrm{const}.

NIFTy takes advantage of this formulation in several ways:

1) All prior degrees of freedom have unit covariance, which improves the condition number of operators that need to be inverted.

2) The amplitude operator can be regarded as part of the response, :math:`{R'=R\,A}`.
   In general, more sophisticated responses can be obtained by combining simpler operators.

3) The response can be non-linear, e.g. :math:`{R'(s)=R \exp(A\,\xi)}`, see `demos/getting_started_2.py`.

4) The amplitude operator may depend on further parameters, e.g. :math:`A=A(\tau)=F\, \widehat{e^\tau}` represents an amplitude operator with a positive definite, unknown spectrum.
   The log-amplitude field :math:`{\tau}` is modelled with the help of an integrated Wiener process in order to impose some (user-defined degree of) spectral smoothness.

5) NIFTy calculates the gradient of the information Hamiltonian and the Fisher information metric with respect to all unknown parameters, here :math:`{\xi}` and :math:`{\tau}`, by automatic differentiation.
   The gradients are used for MAP estimates, and the Fisher matrix is required in addition to the gradient by Metric Gaussian Variational Inference (MGVI), which is available in NIFTy as well.
   MGVI is an implicit operator extension of Automatic Differentiation Variational Inference (ADVI).

The reconstruction of a non-Gaussian signal with unknown covariance from a non-trivial (tomographic) response is demonstrated in `demos/getting_started_3.py`.
Here, the uncertainty of the field and the power spectrum of its generating process are probed via posterior samples provided by the MGVI algorithm.

+----------------------------------------------------+
| **Output of tomography demo getting_started_3.py** |
+----------------------------------------------------+
| .. image:: ../images/getting_started_3_setup.png   |
|                                                    |
+----------------------------------------------------+
| Non-Gaussian signal field,                         |
| data backprojected into the image domain, power    |
| spectrum of underlying Gausssian process.          |
+----------------------------------------------------+
| .. image:: ../images/getting_started_3_results.png |
|                                                    |
+----------------------------------------------------+
| Posterior mean field signal                        |
| reconstruction, its uncertainty, and the power     |
| spectrum of the process for different posterior    |
| samples in comparison to the correct one (thick    |
| orange line).                                      |
+----------------------------------------------------+

Maximum a Posteriori
--------------------

One popular field estimation method is Maximum a Posteriori (MAP).

It only requires minimizing the information Hamiltonian, e.g. by a gradient descent method that stops when

.. math::

    \frac{\partial \mathcal{H}(d,\xi)}{\partial \xi} = 0.

NIFTy8 automatically calculates the necessary gradient from a generative model of the signal and the data and uses this to minimize the Hamiltonian.

However, MAP often provides unsatisfactory results in cases of deep hierarchical Bayesian networks.
The reason for this is that MAP ignores the volume factors in parameter space, which are not to be neglected in deciding whether a solution is reasonable or not.
In the high dimensional setting of field inference these volume factors can differ by large ratios.
A MAP estimate, which is only representative for a tiny fraction of the parameter space, might be a poorer choice (with respect to an error norm) compared to a slightly worse location with slightly lower posterior probability, which, however, is associated with a much larger volume (of nearby locations with similar probability).

This causes MAP signal estimates to be more prone to overfitting the noise as well as to perception thresholds than methods that take volume effects into account.
