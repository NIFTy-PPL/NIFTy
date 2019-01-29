Discretization and Volume in NIFTy
==================================

.. note:: Some of this discussion is rather technical and may be skipped in a first read-through.

Setup
.....

IFT employs stochastic processes to model distributions over function spaces, in particular Gaussian processes :math:`s \sim \mathcal{G}(s,k)` where :math:`k` denotes the covariance function.
The domain of the fields, and hence :math:`k`, is given by a Riemannian manifold :math:`(\mathcal{M},g)`, where :math:`g` denotes a Riemannian metric.
Fields are defined to be scalar functions on the manifold, living in the function space :math:`\mathcal{F}(\mathcal{M})`.
Unless we find ourselves in the lucky situation that we can solve for the posterior statistics of interest analytically, we need to apply numerical methods.
This is where NIFTy comes into play.

.. figure:: images/inference.png
    :width: 80%
    :align: center

    Figure 1: Sketch of the various spaces and maps involved in the inference process.

A typical setup for inference of such signals using NIFTy is shown in figure 1.
We start with a continuous signal :math:`s \in \mathcal{S}`, defined in some function space :math:`\mathcal{S} := \mathcal{F}(\mathcal{M})` over a manifold :math:`(\mathcal{M},g)` with metric :math:`g`.
This is measured by some instrument, e.g. a telescope.
The measurement produces data in an unstructured data space :math:`\mathcal{D}` via a known response function :math:`R : \mathcal{S} \rightarrow \mathcal{D}` and involves noise :math:`\mathcal{D} \ni n \sim \mathcal{N}(n | 0, N)` with known covariance matrix :math:`N`.
In the case of additive noise, the result of the measurement is given by

.. math::

    d = R(s) + n \, .



Discretisation and index notation
.................................

To compute anything numerically, we first need to represent the problem in finite dimensions.
As for stochastic processes, several discretisations of :math:`\mathcal{S}` like collocation methods, expansion into orthogonal polynomials, etc. can be used (see [6]_, [7]_ for an overview and further information about their reliability).
In particular, NIFTy uses the midpoint method as reviewed in section 2.1 in [6]_ and Fourier expansion.

Without going into the details, discretisation methods basically introduce a finite set of basis functions :math:`\{\phi_i\}_{i\in \mathcal{I}}`, where :math:`\mathcal{I}` denotes a generic index set with :math:`|\mathcal{I}| = N` being the chosen discretisation dimension.
Any Riemannian manifold :math:`(\mathcal{M},g)` is equipped with a canonical scalar product given by

.. math::

    \left< a , b \right>_{\mathcal{M}} = \int_{\mathcal{M}} \mathrm{d}x \, \sqrt{|g|} \, a(x) \, b(x) \, .

Projection to the finite basis :math:`\{\phi_i\}_{i\in \mathcal{I}}` is then given by

.. math::

    f^i = v^{ij} \, \left< f , \phi_j \right>_{\mathcal{M}} \,

where the Einstein summation convention is assumed and we defined the volume metric

.. math::
   v_{ij} = \left< \phi_i , \phi_j \right>_{\mathcal{M}} \, ,

along with its inverse, :math:`v^{ij}`, satisfying :math:`v^{ij}v_{jk} = \delta^i_k`.

Obviously, the basis :math:`\{\phi_i\}_{i\in \mathcal{I}}` needs to be chosen s.th. the volume metric is invertible, otherwise we run into trouble.
Volume factors are encoded into the :math:`v_{ij}`.
For specific choices of the basis :math:`\{\phi_i\}_{i\in \mathcal{I}}`, e.g. indicator functions in the case of a pixelation, the entries of :math:`v_{ij}` are indeed just the volumes of the elements.
Lowering and raising indices works with :math:`v_{ij}` and :math:`v^{ij}` just as usual.

After projection, any function :math:`f \in \mathcal{S}` is represented by its approximation :math:`\hat{f} \in \hat{\mathcal{S}} \simeq \mathbb{R}^N`, where

.. math::

    \hat{f} = f^i\,\phi_i \, ,

which defines an embedding :math:`\hat{\mathcal{S}} \hookrightarrow \mathcal{S} = \mathcal{F}(\mathcal{M})`.

**Changes of bases** are performed by reapproximating the :math:`\{\phi_i\}_{i\in \mathcal{I}}` in terms of another basis :math:`\{\phi'_i\}_{i\in \mathcal{I'}}` :

.. math::

    \phi_i \approx (\phi_i)^j \, \phi'_j \, =: \beta_i^{\,j} \, \phi'_j \, ,

which in general implies additional loss of information unless the two bases are compatible, i.e. encode the same information.
The latter is e.g. true for regular collocation grids on tori and the associated cropped Fourier series.
The discrete Fourier transform then maps between those bases without loss of information.

**Discretisation of operators** works in the same way by expansion.
For illustration purposes, let :math:`A: \mathcal{S} \rightarrow \mathcal{S}` be a not necessarily linear operator.
The result of its action on functions :math:`s` is known and may be expanded in :math:`\{\phi_i\}_{i\in \mathcal{I}}`, i.e.

.. math::
   A[s] = (A[s])^k \, \phi_k \, ,

where the domain of the operator may be restricted to the image of the embedding given above.
Integrals can now be written as

.. math::

    \left< s , A[t] \right>_{\mathcal{M}} \approx s^i \left< \phi_i , \phi_j \right>_{\mathcal{M}} (A[t])^j \equiv s^i \, v_{ij} \, (A[t])^j \, ,

where the appearence of the volume metric can be hidden by lowering the first index of the operator,

.. math::

    (A[w])_k := v_{km} \, (A[w])^m \, .

Hence, the volume metric needs not to be carried along if the operators are defined in this fashion right from the start.
Linear operators mapping several functions to another function are completly specified by their action on a given basis, and we define

.. math::

    A^k_{\,\,mn\ldots} := (A[\phi_m,\phi_n,\ldots])^k \, .

If :math:`A` is a (linear) integral operator defined by a kernel :math:`\tilde{A}: \mathcal{M} \times \cdots \mathcal{M} \rightarrow \mathbb{R}`, its components due to :math:`\{\phi_i\}_{i\in \mathcal{I}}` are given by

.. math::

    A^k_{\,\,ij\ldots}
    &= v^{km} \, \left< \phi_m, A[\phi_i,\phi_j,\ldots] \right>_{\mathcal{M}} \\
    &= v^{km} \, \int_{\mathcal{M}} \mathrm{d} x\,\sqrt{|g|}\,\left(\prod_{n}^{|\{ij\ldots\}|}\int_{\mathcal{M}} \mathrm{d} y_n \, \sqrt{|g|}\right) \,\,\phi_m(x)\, \tilde{A}(x,y_1,y_2,\ldots)\, \phi_i(y_1) \, \phi_j(y_2) \cdots \, .

.. [6] Bruno Sudret and Armen Der Kiureghian (2000), "Stochastic Finite Element Methods and Reliability: A State-of-the-Art Report"
.. [7] Dongbin Xiu (2010), "Numerical methods for stochastic computations", Princeton University Press.

Resolution and self-consistency
...............................

Looking at figure 1, we see that the there are two response operators:
On the one hand, there is the actual response :math:`R: \mathcal{S} \rightarrow \mathcal{D}` of the instrument used for measurement, mapping the actual signal to data.
On the other hand, there is a discretised response :math:`\hat{R}: \hat{\mathcal{S}} \rightarrow \mathcal{D}`, mapping from the discretised space to data.
Apparently, the discretisation and the discretised response need to satisfy a self-consistency equation, given by

.. math::
    R = \hat{R} \circ D \, .

An obvious corrollary is that different discretisations :math:`D, D'` with resulting discretised responses :math:`\hat{R}, \hat{R}'` will need to satisfy

.. math::
    \hat{R} \circ D = \hat{R}' \circ D' \, .

NIFTy is implemented such that in order to change resolution, only the line of code defining the space needs to be altered.
It automatically takes care of depended structures like volume factors, discretised operators and responses.
A visualisation of this can be seen in figure 2 and 3, which displays the MAP inference of a signal at various resolutions.

.. figure:: images/converging_discretization.png
    :scale: 80%
    :align: center

    Figure 3: Inference result converging at high resolution.


Implementation in NIFTy
-----------------------

.. currentmodule:: nifty5

Most codes in NIFTy will contain the description of a measurement process,
or more generally, a log-likelihood.
This log-likelihood is necessarily a map from the quantity of interest (a field) to a real number.
The likelihood has to be unitless because it is a log-probability and should not scale with resolution.
Often, likelihoods contain integrals over the quantity of interest :math:`s`, which have to be discretized, e.g. by a sum

.. math::

    \int_\Omega \text{d}x\, s(x) \approx \sum_i s_i\int_{\Omega_i}\text{d}x\, 1

Here the domain of the integral :math:`\Omega = \dot{\bigcup_q} \; \Omega_i` is the disjoint union over smaller :math:`\Omega_i`, e.g. the pixels of the space, and :math:`s_i` is the discretized field value on the :math:`i`-th pixel.
This introduces the weighting :math:`V_i=\int_{\Omega_i}\text{d}x\, 1`, also called the volume factor, a property of the space.
NIFTy aids you in constructing your own likelihood by providing methods like :func:`~field.Field.weight`, which weights all pixels of a field with its corresponding volume.
An integral over a :class:`~field.Field` :code:`s` can be performed by calling :code:`s.weight(1).sum()`, which is equivalent to :code:`s.integrate()`.
Volume factors are also applied automatically in the following places:

 - :class:`~operators.harmonic_operators.FFTOperator` as well as all other harmonic operators. Here the zero mode of the transformed field is the integral over the original field, thus the whole field is weighted once.
 - some response operators, such as the :class:`~library.los_response.LOSResponse`. In this operator a line integral is descritized, so a 1-dimensional volume factor is applied.
 - In :class:`~library.correlated_fields.CorrelatedField` as well :class:`~library.correlated_fields.MfCorrelatedField`, the field is multiplied by the square root of the total volume in configuration space. This ensures that the same field reconstructed over a larger domain has the same variance in position space in the limit of infinite resolution. It also ensures that power spectra in NIFTy behave according to the definition of a power spectrum, namely the power of a k-mode is the expectation of the k-mode square, divided by the volume of the space.

Note that in contrast to some older versions of NIFTy, the dot product of fields does not apply a volume factor

.. math::
  s^\dagger t = \sum_i s_i^* t_i .

If this dot product is supposed to be invariant under changes in resolution, then either :math:`s_i` or :math:`t_i` has to decrease as the number of pixels increases, or more specifically, one of the two fields has to be an extensive quantity while the other has to be intensive.
One can make this more explicit by denoting intensive quantities with upper index and extensive quantities with lower index

.. math::
  s^\dagger t =  (s^*)^i t_i

where we used Einstein sum convention.
This notation connects to the theoretical discussion before.
One of the field has to have the volume metric already incorperated to assure the continouum limit works.
When building statistical models, all indices will end up matching this upper-lower convention automatically, e.g. for a Gaussian log-likelihood :math:`L` we have

.. math::
  L = \frac{1}{2}s^i \left(S^{-1}\right)_{ij} s^j

with

.. math::
  \left(S^{-1}\right)_{ij} = \left(S^{kl}\right)_ij^{-1} = \left(\left<(s^*)^ks^l\right>\right)^{-1})_{ij}\ .

Thus the covariance matrix :math:`S` will ensure that the whole likelihood expression does not scale with resolution.
**This upper-lower index convention is not coded into NIFTy**, in order to not reduce user freedom.
One should however have this in mind when constructing algorithms in order to ensure resolution independence.

Note that while the upper-lower index convention ensures resolution independence, this does not automatically fix the pixilization.
