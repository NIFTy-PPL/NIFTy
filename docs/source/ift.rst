IFT -- Information Field Theory
===============================

Theoretical Background
----------------------


`Information Field Theory <http://www.mpa-garching.mpg.de/ift/>`_ [1]_  (IFT) is information theory, the logic of reasoning under uncertainty, applied to fields.
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

.. tip:: *In-a-nutshell introductions to information field theory* can be found in [2]_, [3]_, [4]_, and [5]_, with the latter probably being the most didactical.

.. [1] T.A. Enßlin et al. (2009), "Information field theory for cosmological perturbation reconstruction and nonlinear signal analysis", PhysRevD.80.105005, 09/2009; `[arXiv:0806.3474] <http://www.arxiv.org/abs/0806.3474>`_

.. [2] T.A. Enßlin (2013), "Information field theory", proceedings of MaxEnt 2012 -- the 32nd International Workshop on Bayesian Inference and Maximum Entropy Methods in Science and Engineering; AIP Conference Proceedings, Volume 1553, Issue 1, p.184; `[arXiv:1301.2556] <http://arxiv.org/abs/1301.2556>`_

.. [3] T.A. Enßlin (2014), "Astrophysical data analysis with information field theory", AIP Conference Proceedings, Volume 1636, Issue 1, p.49; `[arXiv:1405.7701] <http://arxiv.org/abs/1405.7701>`_

.. [4] Wikipedia contributors (2018), `"Information field theory" <https://en.wikipedia.org/w/index.php?title=Information_field_theory&oldid=876731720>`_, Wikipedia, The Free Encyclopedia.

.. [5] T.A. Enßlin (2019), "Information theory for fields", accepted by Annalen der Physik; `[DOI] <https://doi.org/10.1002/andp.201800127>`_, `[arXiv:1804.03350] <http://arxiv.org/abs/1804.03350>`_


Discretized continuum
---------------------

The representation of fields that are mathematically defined on a continuous space in a finite computer environment is a common necessity.
The goal hereby is to preserve the continuum limit in the calculus in order to ensure a resolution independent discretization.

+-----------------------------+-----------------------------+
| .. image:: images/42vs6.png | .. image:: images/42vs9.png |
|     :width:  100 %          |     :width:  100 %          |
+-----------------------------+-----------------------------+

Any partition of the continuous position space :math:`\Omega` (with volume :math:`V`) into a set of :math:`Q` disjoint, proper subsets :math:`\Omega_q` (with volumes :math:`V_q`) defines a pixelization,

.. math::

    \Omega &\quad=\quad \dot{\bigcup_q} \; \Omega_q \qquad \mathrm{with} \qquad q \in \{1,\dots,Q\} \subset \mathbb{N}
    , \\
    V &\quad=\quad \int_\Omega \mathrm{d}x \quad=\quad \sum_{q=1}^Q \int_{\Omega_q} \mathrm{d}x \quad=\quad \sum_{q=1}^Q V_q
    .

Here the number :math:`Q` characterizes the resolution of the pixelization and the continuum limit is described by :math:`Q \rightarrow \infty` and :math:`V_q \rightarrow 0` for all :math:`q \in \{1,\dots,Q\}` simultaneously.
Moreover, the above equation defines a discretization of continuous integrals, :math:`\int_\Omega \mathrm{d}x \mapsto \sum_q V_q`.

Any valid discretization scheme for a field :math:`{s}` can be described by a mapping,

.. math::

    s(x \in \Omega_q) \quad\mapsto\quad s_q \quad=\quad \int_{\Omega_q} \mathrm{d}x \; w_q(x) \; s(x)
    ,

if the weighting function :math:`w_q(x)` is chosen appropriately.
In order for the discretized version of the field to converge to the actual field in the continuum limit, the weighting functions need to be normalized in each subset; i.e., :math:`\forall q: \int_{\Omega_q} \mathrm{d}x \; w_q(x) = 1`.
Choosing such a weighting function that is constant with respect to :math:`x` yields

.. math::

    s_q = \frac{\int_{\Omega_q} \mathrm{d}x \; s(x)}{\int_{\Omega_q} \mathrm{d}x} = \left< s(x) \right>_{\Omega_q}
    ,

which corresponds to a discretization of the field by spatial averaging.
Another common and equally valid choice is :math:`w_q(x) = \delta(x-x_q)`, which distinguishes some position :math:`x_q \in \Omega_q`, and evaluates the continuous field at this position,

.. math::

    s_q \quad=\quad \int_{\Omega_q} \mathrm{d}x \; \delta(x-x_q) \; s(x) \quad=\quad s(x_q)
    .

In practice, one often makes use of the spatially averaged pixel position, :math:`x_q = \left< x \right>_{\Omega_q}`.
If the resolution is high enough to resolve all features of the signal field :math:`{s}`, both of these discretization schemes approximate each other, :math:`\left< s(x) \right>_{\Omega_q} \approx s(\left< x \right>_{\Omega_q})`, since they approximate the continuum limit by construction.
(The approximation of :math:`\left< s(x) \right>_{\Omega_q} \approx s(x_q \in \Omega_q)` marks a resolution threshold beyond which further refinement of the discretization reveals no new features; i.e., no new information content of the field :math:`{s}`.)

All operations involving position integrals can be normalized in accordance with the above definitions.
For example, the scalar product between two fields :math:`{s}` and :math:`{u}` is defined as

.. math::

    {s}^\dagger {u} \quad=\quad \int_\Omega \mathrm{d}x \; s^*(x) \; u(x) \quad\approx\quad \sum_{q=1}^Q V_q^{\phantom{*}} \; s_q^* \; u_q^{\phantom{*}}
    ,

where :math:`\dagger` denotes adjunction and :math:`*` complex conjugation.
Since the above approximation becomes an equality in the continuum limit, the scalar product is independent of the pixelization scheme and resolution, if the latter is sufficiently high.

The above line of argumentation analogously applies to the discretization of operators.
For a linear operator :math:`{A}` acting on some field :math:`{s}` as :math:`{A} {s} = \int_\Omega \mathrm{d}y \; A(x,y) \; s(y)`, a matrix representation discretized with constant weighting functions is given by

.. math::

    A(x \in \Omega_p, y \in \Omega_q) \quad\mapsto\quad A_{pq} \quad=\quad \frac{\iint_{\Omega_p \Omega_q} \mathrm{d}x \, \mathrm{d}y \; A(x,y)}{\iint_{\Omega_p \Omega_q} \mathrm{d}x \, \mathrm{d}y} \quad=\quad \big< \big< A(x,y) \big>_{\Omega_p} \big>_{\Omega_q}
    .

The proper discretization of spaces, fields, and operators, as well as the normalization of position integrals, is essential for the conservation of the continuum limit.
Their consistent implementation in NIFTy allows a pixelization independent coding of algorithms.

Free Theory & Implicit Operators
--------------------------------

A free IFT appears when the signal field :math:`{s}` and the noise :math:`{n}` of the data :math:`{d}` are independent, zero-centered Gaussian processes of kown covariances :math:`{S}` and :math:`{N}`, respectively,

.. math::

    \mathcal{P}(s,n) = \mathcal{G}(s,S)\,\mathcal{G}(n,N),

and the measurement equation is linear in both signal and noise,

.. math::

    d= R\, s + n,

with :math:`{R}` being the measurement response, which maps the continous signal field into the discrete data space.

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
The advantage of implicit operators to explicit matrices is the reduced memory requirements.
The reconstruction of only a Megapixel image would otherwithe require the storage and processing of matrices with sizes of several Terabytes.
Larger images could not be dealt with due to the quadratic memory requirements of explicit operator representations.

The demo codes `demos/getting_started_1.py` and `demos/Wiener_Filter.ipynb` illustrate this.


Generative Models
-----------------

For more sophisticated measurement situations, involving non-linear measuremnts, unknown covariances, calibration constants and the like, it is recommended to formulate those as generative models for which NIFTy provides powerful inference algorithms.

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
   In general, more sophisticated responses can be constructed out of the composition of simpler operators.

3) The response can be non-linear, e.g. :math:`{R'(s)=R \exp(A\,\xi)}`, see `demos/getting_started_2.py`.

4) The amplitude operator may dependent on further parameters, e.g. :math:`A=A(\tau)= F\, \widehat{e^\tau}` represents an amplitude operator with a positive definite, unknown spectrum defined in the Fourier domain.
   The amplitude field :math:`{\tau}` would get its own amplitude operator, with a cepstrum (spectrum of a log spectrum) defined in quefrency space (harmonic space of a logarithmically binned harmonic space) to regularize its degrees of freedom by imposing some (user-defined degree of) spectral smoothness.

5) NIFTy calculates the gradient of the information Hamiltonian and the Fisher information metric with respect to all unknown parameters, here :math:`{\xi}` and :math:`{\tau}`, by automatic differentiation.
   The gradients are used for MAP and HMCF estimates, and the Fisher matrix is required in addition to the gradient by Metric Gaussian Variational Inference (MGVI), which is available in NIFTy as well.
   MGVI is an implicit operator extension of Automatic Differentiation Variational Inference (ADVI).

The reconstruction of a non-Gaussian signal with unknown covariance from a non-trivial (tomographic) response is demonstrated in `demos/getting_started_3.py`.
Here, the uncertainty of the field and the power spectrum of its generating process are probed via posterior samples provided by the MGVI algorithm.

+----------------------------------------------------+
| **Output of tomography demo getting_started_3.py** |
+----------------------------------------------------+
| .. image:: images/getting_started_3_setup.png      |
|     :width:  50 %                                  |
+----------------------------------------------------+
| Non-Gaussian signal field,                         |
| data backprojected into the image domain, power    |
| spectrum of underlying Gausssian process.          |
+----------------------------------------------------+
| .. image:: images/getting_started_3_results.png    |
|     :width:  50 %                                  |
+----------------------------------------------------+
| Posterior mean field signal                        |
| reconstruction, its uncertainty, and the power     |
| spectrum of the process for different posterior    |
| samples in comparison to the correct one (thick    |
| orange line).                                      |
+----------------------------------------------------+

Maximim a Posteriori
--------------------

One popular field estimation method is Maximim a Posteriori (MAP).

It only requires to minimize the information Hamiltonian, e.g by a gradient descent method that stops when

.. math::

    \frac{\partial \mathcal{H}(d,\xi)}{\partial \xi} = 0.

NIFTy5 automatically calculates the necessary gradient from a generative model of the signal and the data and to minimize the Hamiltonian.

However, MAP often provides unsatisfactory results in cases of deep hirachical Bayesian networks.
The reason for this is that MAP ignores the volume factors in parameter space, which are not to be neglected in deciding whether a solution is reasonable or not.
In the high dimensional setting of field inference these volume factors can differ by large ratios.
A MAP estimate, which is only representative for a tiny fraction of the parameter space, might be a poorer choice (with respect to an error norm) compared to a slightly worse location with slightly lower posterior probability, which, however, is associated with a much larger volume (of nearby locations with similar probability).

This causes MAP signal estimates to be more prone to overfitting the noise as well as to perception thresholds than methods that take volume effects into account.


Variational Inference
---------------------

One method that takes volume effects into account is Variational Inference (VI).
In VI, the posterior :math:`\mathcal{P}(\xi|d)` is approximated by a simpler, parametrized distribution, often a Gaussian :math:`\mathcal{Q}(\xi)=\mathcal{G}(\xi-m,D)`.
The parameters of :math:`\mathcal{Q}`, the mean :math:`m` and its covariance :math:`D` are obtained by minimization of an appropriate information distance measure between :math:`\mathcal{Q}` and :math:`\mathcal{P}`.
As a compromise between being optimal and being computationally affordable, the variational Kullback-Leibler (KL) divergence is used:

.. math::

    \mathrm{KL}(m,D|d)= \mathcal{D}_\mathrm{KL}(\mathcal{Q}||\mathcal{P})=
    \int \mathcal{D}\xi \,\mathcal{Q}(\xi) \log \left( \frac{\mathcal{Q}(\xi)}{\mathcal{P}(\xi)} \right)

Minimizing this with respect to all entries of the covariance :math:`D` is unfeasible for fields.
Therefore, Metric Gaussian Variational Inference (MGVI) approximates the precision matrix at the location of the current mean :math:`M=D^{-1}` by the Bayesian Fisher information metric,

.. math::

    M \approx \left\langle \frac{\partial \mathcal{H}(d,\xi)}{\partial \xi} \, \frac{\partial \mathcal{H}(d,\xi)}{\partial \xi}^\dagger \right\rangle_{(d,\xi)}.

In practice the average is performed over :math:`\mathcal{P}(d,\xi)\approx \mathcal{P}(d|\xi)\,\delta(\xi-m)` by evaluating the expression at the current mean :math:`m`.
This results in a Fisher information metric of the likelihood evaluated at the mean plus the prior information metric.
Therefore we will only have to infer the mean of the approximate distribution.
The only term within the KL-divergence that explicitly depends on it is the Hamiltonian of the true problem averaged over the approximation:

.. math::

    \mathrm{KL}(m|d) \;\widehat{=}\;
    \left\langle  \mathcal{H}(\xi,d)    \right\rangle_{\mathcal{Q}(\xi)},

where :math:`\widehat{=}` expresses equality up to irrelvant (here not :math:`m`-dependent) terms.

Thus, only the gradient of the KL is needed with respect to this, which can be expressed as

.. math::

    \frac{\partial \mathrm{KL}(m|d)}{\partial m} = \left\langle \frac{\partial \mathcal{H}(d,\xi)}{\partial \xi}  \right\rangle_{\mathcal{G}(\xi-m,D)}.

We stochastically estimate the KL-divergence and gradients with a set of samples drawn from the approximate posterior distribution.
The particular structure of the covariance allows us to draw independent samples solving a certain system of equations.
This KL-divergence for MGVI is implemented in the class MetricGaussianKL within NIFTy5.


The demo `getting_started_3.py` for example not only infers a field this way, but also the power spectrum of the process that has generated the field.
The cross-correlation of field and power spectrum is taken care of in this process.
Posterior samples can be obtained to study this cross-correlation.

It should be noted that MGVI, as any VI method, can typically only provide a lower bound on the variance.
