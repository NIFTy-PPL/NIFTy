Approximate Inference
=====================

In Variational Inference (VI), the posterior :math:`\mathcal{P}(\xi|d)` is approximated by a simpler, parametrized distribution, often a Gaussian :math:`\mathcal{Q}(\xi)=\mathcal{G}(\xi-m,D)`.
The parameters of :math:`\mathcal{Q}`, the mean :math:`m` and its covariance :math:`D` are obtained by minimization of an appropriate information distance measure between :math:`\mathcal{Q}` and :math:`\mathcal{P}`.
As a compromise between being optimal and being computationally affordable, the variational Kullback-Leibler (KL) divergence is used:

.. math::

    \mathrm{KL}(m,D|d)= \mathcal{D}_\mathrm{KL}(\mathcal{Q}||\mathcal{P})=
    \int \mathcal{D}\xi \,\mathcal{Q}(\xi) \log \left( \frac{\mathcal{Q}(\xi)}{\mathcal{P}(\xi)} \right)

NIFTy features two main alternatives for variational inference: Metric Gaussian Variational Inference (MGVI) and geometric Variational Inference (geoVI).
A visual comparison of the MGVI and GeoVI algorithm can be found in `variational_inference_visualized.py <https://gitlab.mpcdf.mpg.de/ift/nifty/-/blob/NIFTy_8/demos/variational_inference_visualized.py>`_.


Metric Gaussian Variational Inference (MGVI)
--------------------------------------------

Minimizing the KL divergence with respect to all entries of the covariance :math:`D` is unfeasible for fields.
Therefore, Metric Gaussian Variational Inference (MGVI, [1]_) approximates the posterior precision matrix :math:`D^{-1}` at the location of the current mean :math:`m` by the Bayesian Fisher information metric,

.. math::

    M \approx \left\langle \frac{\partial \mathcal{H}(d,\xi)}{\partial \xi} \, \frac{\partial \mathcal{H}(d,\xi)}{\partial \xi}^\dagger \right\rangle_{(d,\xi)}.

In practice the average is performed over :math:`\mathcal{P}(d,\xi)\approx \mathcal{P}(d|\xi)\,\delta(\xi-m)` by evaluating the expression at the current mean :math:`m`.
This results in a Fisher information metric of the likelihood evaluated at the mean plus the prior information metric.
Therefore we will only have to infer the mean of the approximate distribution.
The only term within the KL-divergence that explicitly depends on it is the Hamiltonian of the true problem averaged over the approximation:

.. math::

    \mathrm{KL}(m|d) \;\widehat{=}\;
    \left\langle  \mathcal{H}(\xi,d)    \right\rangle_{\mathcal{Q}(\xi)},

where :math:`\widehat{=}` expresses equality up to irrelevant (here not :math:`m`-dependent) terms.

Thus, only the gradient of the KL is needed with respect to this, which can be expressed as

.. math::

    \frac{\partial \mathrm{KL}(m|d)}{\partial m} = \left\langle \frac{\partial \mathcal{H}(d,\xi)}{\partial \xi}  \right\rangle_{\mathcal{G}(\xi-m,D)}.

We stochastically estimate the KL-divergence and gradients with a set of samples drawn from the approximate posterior distribution.
The particular structure of the covariance allows us to draw independent samples solving a certain system of equations.
This KL-divergence for MGVI is implemented by
:func:`~nifty8.minimization.kl_energies.SampledKLEnergy` within NIFTy8.

Note that MGVI typically provides only a lower bound on the variance.



Geometric Variational Inference (geoVI)
---------------------------------------

For non-linear posterior distributions :math:`\mathcal{P}(\xi|d)` an approximation with a Gaussian :math:`\mathcal{Q}(\xi)` in the coordinates :math:`\xi` is sub-optimal, as higher order interactions are ignored.
A better approximation can be achieved by constructing a coordinate system :math:`y = g\left(\xi\right)` in which the posterior is close to a Gaussian, and perform VI with a Gaussian :math:`\mathcal{Q}(y)` in these coordinates.
This approach is called Geometric Variational Inference (geoVI).
It is discussed in detail in [2]_.

One useful coordinate system is obtained in case the metric :math:`M` of the posterior can be expressed as the pullback of the Euclidean metric by :math:`g`:

.. math::

    M = \left(\frac{\partial g}{\partial \xi}\right)^T \frac{\partial g}{\partial \xi} \ .

In general, such a transformation exists only locally, i.e. in a neighbourhood of some expansion point :math:`\bar{\xi}`, denoted as :math:`g_{\bar{\xi}}\left(\xi\right)`.
Using :math:`g_{\bar{\xi}}`, the GeoVI scheme uses a zero mean, unit Gaussian :math:`\mathcal{Q}(y) = \mathcal{G}(y, 1)` approximation.
It can be expressed in :math:`\xi` coordinates via the pushforward by the inverse transformation :math:`\xi = g_{\bar{\xi}}^{-1}(y)`:

.. math::

    \mathcal{Q}_{\bar{\xi}}(\xi) = \left(g_{\bar{\xi}}^{-1} * \mathcal{Q}\right)(\xi) = \int \delta\left(\xi - g_{\bar{\xi}}^{-1}(y)\right) \ \mathcal{G}(y, 1) \ \mathcal{D}y \ ,

where :math:`\delta` denotes the Kronecker-delta.

GeoVI obtains the optimal expansion point :math:`\bar{\xi}` such that :math:`\mathcal{Q}_{\bar{\xi}}` matches the posterior as good as possible.
Analogous to the MGVI algorithm, :math:`\bar{\xi}` is obtained by minimization of the KL-divergence between :math:`\mathcal{P}` and :math:`\mathcal{Q}_{\bar{\xi}}` w.r.t. :math:`\bar{\xi}`.
Furthermore the KL is represented as a stochastic estimate using a set of samples drawn from :math:`\mathcal{Q}_{\bar{\xi}}` which is implemented in NIFTy8 via :func:`~nifty8.minimization.kl_energies.SampledKLEnergy` with `minimizer_sampling != None`.


Publications
------------

If you use MGVI or geoVI, the authors of the respective papers would greatly appreciate a citation.

.. [1] J. Knollmüller, T.A. Enßlin, "Metric Gaussian Variational Inference"; `[arXiv:1901.11033] <https://arxiv.org/abs/1901.11033>`_

.. [2] P. Frank, R. Leike, and T.A. Enßlin (2021), "Geometric Variational Inference"; `[arXiv:2105.10470] <https://arxiv.org/abs/2105.10470>`_ `[doi] <https://doi.org/10.3390/e23070853>`_

