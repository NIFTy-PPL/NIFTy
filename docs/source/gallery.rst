Image Gallery
-------------

Transformations & Projections
.............................

.. currentmodule:: nifty

The "Faraday Map" [1]_ in spherical representation on a :py:class:`hp_space` and a :py:class:`gl_space`, their quadrupole projections, the uncertainty of the map, and the angular power spectrum.

+----------------------------+----------------------------+
| .. image:: images/f_00.png | .. image:: images/f_01.png |
|     :width:  90 %          |     :width:  90 %          |
+----------------------------+----------------------------+
| .. image:: images/f_02.png | .. image:: images/f_03.png |
|     :width:  90 %          |     :width:  90 %          |
+----------------------------+----------------------------+
| .. image:: images/f_04.png | .. image:: images/f_05.png |
|     :width:  90 %          |     :width:  70 %          |
+----------------------------+----------------------------+

Gaussian random fields
......................

Statistically homogeneous and isotropic Gaussian random fields drawn from different power spectra.

+----------------------------+----------------------------+
| .. image:: images/t_03.png | .. image:: images/t_04.png |
|     :width:  60 %          |     :width:  70 %          |
+----------------------------+----------------------------+
| .. image:: images/t_05.png | .. image:: images/t_06.png |
|     :width:  60 %          |     :width:  70 %          |
+----------------------------+----------------------------+


Wiener filtering I
..................

Wiener filter reconstruction of Gaussian random signal.

+--------------------------------+--------------------------------+--------------------------------+
| original signal                | noisy data                     | reconstruction                 |
+================================+================================+================================+
| .. image:: images/rg1_s.png    | .. image:: images/rg1_d.png    | .. image:: images/rg1_m.png    |
|     :width:  90 %              |     :width:  90 %              |     :width:  90 %              |
+--------------------------------+--------------------------------+--------------------------------+
| .. image:: images/rg2_s_pm.png | .. image:: images/rg2_d_pm.png | .. image:: images/rg2_m_pm.png |
|     :width:  90 %              |     :width:  90 %              |     :width:  90 %              |
+--------------------------------+--------------------------------+--------------------------------+
| .. image:: images/hp_s.png     | .. image:: images/hp_d.png     | .. image:: images/hp_m.png     |
|     :width:  90 %              |     :width:  90 %              |     :width:  90 %              |
+--------------------------------+--------------------------------+--------------------------------+

Image reconstruction
....................

Image reconstruction of the classic "Moon Surface" image. The original image "Moon Surface" was taken from the `USC-SIPI image database <http://sipi.usc.edu/database/>`_.

+-----------------------------------+-----------------------------------+-----------------------------------+
| .. image:: images/moon_s.png      | .. image:: images/moon_d.png      | .. image:: images/moon_m.png      |
|     :width:  90 %                 |     :width:  90 %                 |     :width:  90 %                 |
+-----------------------------------+-----------------------------------+-----------------------------------+
| .. image:: images/moon_kernel.png | .. image:: images/moon_mask.png   | .. image:: images/moon_sigma.png  |
|     :width:  90 %                 |     :width:  90 %                 |     :width:  90 %                 |
+-----------------------------------+-----------------------------------+-----------------------------------+

Wiener filtering II
...................

Wiener filter reconstruction results for the full and partially blinded data. Shown are the original signal (orange), the reconstruction (green), and :math:`1\sigma`-confidence interval (gray).

+--------------------------------------+--------------------------------------+
| noisy data                           | reconstruction results               |
+======================================+======================================+
| .. image:: images/rg1_d.png          | .. image:: images/rg1_m_err_.png     |
|     :width:  90 %                    |     :width:  90 %                    |
+--------------------------------------+--------------------------------------+
| .. image:: images/rg1_d_gap.png      | .. image:: images/rg1_m_gap_err_.png |
|     :width:  90 %                    |     :width:  90 %                    |
+--------------------------------------+--------------------------------------+

D\ :sup:`3`\ PO -- Denoising, Deconvolving, and Decomposing Photon Observations
...............................................................................

Application of the D\ :sup:`3`\ PO algorithm [2]_ showing the raw photon count data and the denoised, deconvolved, and decomposed reconstruction of the diffuse photon flux.

+--------------------------------------+--------------------------------------+
| .. image:: images/D3PO_data.png      | .. image:: images/D3PO_diffuse.png   |
|     :width:  95 %                    |     :width:  95 %                    |
+--------------------------------------+--------------------------------------+

RESOLVE -- Aperature synthesis imaging in radio astronomy
.........................................................

Signal inference on simulated single-frequency data: reconstruction by CLEAN (using uniform weighting) and by RESOLVE [3]_ (using IFT & NIFTY).

+-------------------------------------+-------------------------------------+-------------------------------------+
| .. image:: images/radio_signal.png  | .. image:: images/radio_CLEAN.png   | .. image:: images/radio_RESOLVE.png |
|     :width:  90 %                   |     :width:  90 %                   |     :width:  90 %                   |
+-------------------------------------+-------------------------------------+-------------------------------------+

D\ :sup:`3`\ PO -- light
........................

Inference of the mock distribution of some species across Australia exploiting geospatial correlations in a (strongly) simplified scenario [4]_.

+--------------------------------+--------------------------------+--------------------------------+
| .. image:: images/au_data.png  | .. image:: images/au_map.png   | .. image:: images/au_error.png |
|     :width:  90 %              |     :width:  90 %              |     :width:  90 %              |
+--------------------------------+--------------------------------+--------------------------------+

NIFTY meets Lensing
...................

Signal reconstruction for a simulated image that has undergone strong gravitational lensing. Without *a priori* knowledge of the signal covariance :math:`S`, a common approach rescaling the  `Laplace-Operator <http://de.wikipedia.org/wiki/Laplace-Operator>`_ and IFT's `"critical" filter <./demo_excaliwir.html#critical-wiener-filtering>`_ are compared.

+--------------------------------+--------------------------------+--------------------------------+--------------------------------+
| .. image:: images/lens_s0.png  | .. image:: images/lens_d0.png  | .. image:: images/lens_m1.png  | .. image:: images/lens_m2.png  |
|     :width:  80 %              |     :width:  80 %              |     :width:  80 %              |     :width:  80 %              |
|                                |                                |                                |                                |
|                                |                                | .. math::                      | .. math::                      |
|                                |                                |     S(x,y) &=                  |     S(x,y) &=                  |
|                                |                                |         \lambda \: \Delta^{-1} |         S(|x-y|)               |
|                                |                                |         \\ \equiv              |         \\ \equiv              |
|                                |                                |     S(k,l) &= \delta(k-l)      |     S(k,l) &= \delta(k-l)      |
|                                |                                |         \: \lambda \: k^{-2}   |         \: P(k)                |
+--------------------------------+--------------------------------+--------------------------------+--------------------------------+

.. [1] N. Oppermann et. al., "An improved map of the Galactic Faraday sky", Astronomy & Astrophysics, vol. 542, id. A93, p. 14, see also the `project homepage <http://www.mpa-garching.mpg.de/ift/faraday/>`_

.. [2] M. Selig et. al., "Denoising, Deconvolving, and Decomposing Photon Observations", submitted to Astronomy & Astrophysics, 2013; `arXiv:1311.1888 <http://www.arxiv.org/abs/1311.1888>`_

.. [3] H. Junklewitz et. al., "RESOLVE: A new algorithm for aperture synthesis imaging of extended emission in radio astronomy", submitted to Astronomy & Astrophysics, 2013; `arXiv:1311.5282 <http://www.arxiv.org/abs/1311.5282>`_

.. [4] M. Selig, "The NIFTY way of Bayesian signal inference", submitted proceeding of the 33rd MaxEnt, 2013


