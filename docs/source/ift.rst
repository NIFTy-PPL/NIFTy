IFT -- Information Field Theory
===============================

Theoretical Background
----------------------


`Information Field Theory <http://www.mpa-garching.mpg.de/ift/>`_ [1-5]_ (IFT) is information theory, the logic of reasoning under uncertainty, applied to fields. A field can be any quantity defined over some space, e.g. the air temperature over Europe, the magnetic field strength in the Milky Way, or the matter density in the Universe. IFT describes how data and knowledge can be used to infer field properties. Mathematically it is a statistical field theory and exploits many of the tools developed for such. Practically, it is a framework for signal processing and image reconstruction.

IFT is fully Bayesian. How else could infinitely many field degrees of freedom be constrained by finite data?

It can be used without the knowledge of Feynman diagrams. There is a full toolbox of methods. It reproduces many known well working algorithms. This should be reassuring. And, there were certainly previous works in a similar spirit. Anyhow, in many cases IFT provides novel rigorous ways to extract information from data.

.. tip:: An *in-a-nutshell introduction to information field theory* can be found in [2]_.

.. [1] T. Ensslin (2019), "Information theory for fields", accepted by Annalen der Physik; `arXiv:1804.03350 <http://arxiv.org/abs/1804.03350>`_

.. [2] Wikipedia contributors (2018), "Information field theory", Wikipedia, The Free Encyclopedia. <https://en.wikipedia.org/w/index.php?title=Information_field_theory&oldid=876731720>`_

.. [3] T. Ensslin (2014), "Astrophysical data analysis with information field theory", AIP Conference Proceedings, Volume 1636, Issue 1, p.49; `arXiv:1405.7701 <http://arxiv.org/abs/1405.7701>`_

.. [4] T. Ensslin (2013), "Information field theory", proceedings of MaxEnt 2012 -- the 32nd International Workshop on Bayesian Inference and Maximum Entropy Methods in Science and Engineering; AIP Conference Proceedings, Volume 1553, Issue 1, p.184; `arXiv:1301.2556 <http://arxiv.org/abs/1301.2556>`_

.. [5] T. Ensslin et al. (2009), "Information field theory for cosmological perturbation reconstruction and nonlinear signal analysis", PhysRevD.80.105005, 09/2009; `arXiv:0806.3474 <http://www.arxiv.org/abs/0806.3474>`_

Discretized continuum
---------------------

The representation of fields that are mathematically defined on a continuous space in a finite computer environment is a common necessity. The goal hereby is to preserve the continuum limit in the calculus in order to ensure a resolution independent discretization.

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

Here the number :math:`Q` characterizes the resolution of the pixelization and the continuum limit is described by :math:`Q \rightarrow \infty` and :math:`V_q \rightarrow 0` for all :math:`q \in \{1,\dots,Q\}` simultaneously. Moreover, the above equation defines a discretization of continuous integrals, :math:`\int_\Omega \mathrm{d}x \mapsto \sum_q V_q`.

Any valid discretization scheme for a field :math:`{s}` can be described by a mapping,

.. math::

    s(x \in \Omega_q) \quad\mapsto\quad s_q \quad=\quad \int_{\Omega_q} \mathrm{d}x \; w_q(x) \; s(x)
    ,

if the weighting function :math:`w_q(x)` is chosen appropriately. In order for the discretized version of the field to converge to the actual field in the continuum limit, the weighting functions need to be normalized in each subset; i.e., :math:`\forall q: \int_{\Omega_q} \mathrm{d}x \; w_q(x) = 1`. Choosing such a weighting function that is constant with respect to :math:`x` yields

.. math::

    s_q = \frac{\int_{\Omega_q} \mathrm{d}x \; s(x)}{\int_{\Omega_q} \mathrm{d}x} = \left< s(x) \right>_{\Omega_q}
    ,

which corresponds to a discretization of the field by spatial averaging. Another common and equally valid choice is :math:`w_q(x) = \delta(x-x_q)`, which distinguishes some position :math:`x_q \in \Omega_q`, and evaluates the continuous field at this position,

.. math::

    s_q \quad=\quad \int_{\Omega_q} \mathrm{d}x \; \delta(x-x_q) \; s(x) \quad=\quad s(x_q)
    .

In practice, one often makes use of the spatially averaged pixel position, :math:`x_q = \left< x \right>_{\Omega_q}`. If the resolution is high enough to resolve all features of the signal field :math:`{s}`, both of these discretization schemes approximate each other, :math:`\left< s(x) \right>_{\Omega_q} \approx s(\left< x \right>_{\Omega_q})`, since they approximate the continuum limit by construction. (The approximation of :math:`\left< s(x) \right>_{\Omega_q} \approx s(x_q \in \Omega_q)` marks a resolution threshold beyond which further refinement of the discretization reveals no new features; i.e., no new information content of the field :math:`{s}`.)

All operations involving position integrals can be normalized in accordance with the above definitions. For example, the scalar product between two fields :math:`{s}` and :math:`{u}` is defined as

.. math::

    {s}^\dagger {u} \quad=\quad \int_\Omega \mathrm{d}x \; s^*(x) \; u(x) \quad\approx\quad \sum_{q=1}^Q V_q^{\phantom{*}} \; s_q^* \; u_q^{\phantom{*}}
    ,

where :math:`\dagger` denotes adjunction and :math:`*` complex conjugation. Since the above approximation becomes an equality in the continuum limit, the scalar product is independent of the pixelization scheme and resolution, if the latter is sufficiently high.

The above line of argumentation analogously applies to the discretization of operators. For a linear operator :math:`{A}` acting on some field :math:`{s}` as :math:`{A} {s} = \int_\Omega \mathrm{d}y \; A(x,y) \; s(y)`, a matrix representation discretized with constant weighting functions is given by

.. math::

    A(x \in \Omega_p, y \in \Omega_q) \quad\mapsto\quad A_{pq} \quad=\quad \frac{\iint_{\Omega_p \Omega_q} \mathrm{d}x \, \mathrm{d}y \; A(x,y)}{\iint_{\Omega_p \Omega_q} \mathrm{d}x \, \mathrm{d}y} \quad=\quad \big< \big< A(x,y) \big>_{\Omega_p} \big>_{\Omega_q}
    .

The proper discretization of spaces, fields, and operators, as well as the normalization of position integrals, is essential for the conservation of the continuum limit. Their consistent implementation in NIFTY allows a pixelization independent coding of algorithms.

Free Theory & Implicit Operators 
--------------------------------

A free IFT appears when the signal field :math:`{s}` and the noise :math:`{n}` in the data :math:`{d}` are independent, zero-centered Gaussian processes of kown covariances :math:`{S}` and :math:`{N}`, respectively,

.. math::

    \mathcal{P}(s,n) = \mathcal{G}(s,S)\,\mathcal{G}(n,N),

and the measurement equation is linear in both,

.. math::

    d= R\, s + n,

with :math:`{R}` the measurement response, which maps the continous signal field into the discrete data space.

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

the information source. The operation in :math:`{d= D\,R^\dagger N^{-1} d}` is also called the generalized Wiener filter.

NIFTy permits to define the involved operators :math:`{R}`, :math:`{R^\dagger}`, :math:`{S}`, and :math:`{N}` implicitely, as coputer routines that can be applied to vectors, but which do not require the explicit storage of the matrix elements of the operators. 
These implicit operators can be combined into new operators, e.g. to :math:`{D^{-1} = \left( S^{-1} + R^\dagger N^{-1} R\right)^{-1}}`, as well as their inverses, e.g. :math:`{D^{-1} = \left( D^{-1} \right)^{-1}}`.
The invocation of an inverse operator applied to a vector might trigger the execution of a numerical linear algebra solver.

Thus, when NIFTy calculates :math:`{m = D\, j}` it actually solves  :math:`{D^{-1} m = j}` for :math:`{m}` behind the scenes. 

The demo codes demos/getting_started_1.py and demos/Wiener_Filter.ipynb illustrate this.


Generative Models
-----------------

For more complex measurement situations, involving non-linear measuremnts, unknown covariances, calibration constants and the like, it is recommended to formulate those as generative models as NIFTy provides powerful inference algorithms for such.

In a generative model, all known or unknown quantities are described as the results of generative processes, which start with simple probability distributions, like uniform, iid Gaussian, or delta distributions. 

The above free theory case looks as a generative model like the following:

.. math::

    s = A\,\xi

with :math:`{A}` the amplitude operator such that it generates signal field with the correct covariance :math:`{S=A\,A^\dagger}` out of a Gaussian white noise field :math:`{\xi}` with :math:`{\mathcal{P}(\xi)= \mathcal{G}(\xi, \mathbb{1})}`.

The joint information Hamiltonian for the whitened signal field :math:`{\xi}`  reads


.. math::

    \mathcal{H}(d,\xi)= -\log \mathcal{P}(d,s)= \frac{1}{2} \xi^\dagger \mathbb{1} \xi + \frac{1}{2} (d-R\,A\,\xi)^\dagger N^{-1} (d-R\,A\,\xi) + \mathrm{const}.

NIFTy takes advantage of this formulation in several ways: 

1) all prior degrees of freedom have now the same variance
2) the amplitude operator can be regarded as part of the response, :math:`{R'=R\,A}`
3) the response can be made non-linear, e.g. :math:`{R'(s)=R \exp(A\,\xi)}`, see demos/demos/getting_started_2.py
4) the amplitude operator can be made dependent on unknowns as well, e.g. :math:`{A=A(\tau)=\mathrm{FourierTransform}\,\mathrm{DiagonalOperator}(\exp(\tau))}` represents an amplitude model with a flexible Fourier spectrum
5) the gradient of the Hamiltonian and the Fischer information metric with respect to all unknown parameters, here :math:`{\xi} and can be constructed by NIFTy and used for Metric Gaussian Variational Inference.

A demonstration example for reconstructing a non-Gaussian signal with unknown covarinance from a complex (tomographic) response is given by demos/demos/getting_started_2.py .





