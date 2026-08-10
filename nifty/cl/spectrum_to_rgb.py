import numpy as np


class ColorSpaceTools:
    """Static helper methods for color space conversions and CIE standard data."""

    # --- Color space transformations ---

    @staticmethod
    def XYZ_to_xyY(XYZ_values):
        """Convert CIE 1931 XYZ to chromaticity-luminance xyY.

        Undefined (returns zeros) for black pixels where sum(XYZ) = 0.
        """
        denom = np.sum(XYZ_values, axis=-1)[..., np.newaxis]
        safe_denom = np.where(denom > 0, denom, 1.0)
        res = np.where(denom > 0, XYZ_values / safe_denom, 0.0)
        res[..., 2] = XYZ_values[..., 1]
        return res

    @staticmethod
    def xyY_to_XYZ(xyY_values):
        """Convert chromaticity-luminance xyY to CIE 1931 XYZ.

        Undefined (returns zeros) for black pixels where y = 0.
        """
        res = np.empty_like(xyY_values)
        y = xyY_values[..., 1]
        Y = xyY_values[..., 2]
        safe_y = np.where(y > 0, y, 1.0)
        fct = np.where(y > 0, Y / safe_y, 0.0)
        res[..., 0] = xyY_values[..., 0] * fct
        res[..., 1] = Y
        res[..., 2] = (1. - xyY_values[..., 0] - xyY_values[..., 1]) * fct
        return res

    # Stockman & Sharpe 2000 XYZ↔LMS matrices
    _XYZ_to_LMS_mat = np.array([[0.210576, 0.855098, -0.0396983],
                                 [-0.417076, 1.177260, 0.0786283],
                                 [0.0, 0.0, 0.5168350]])

    _LMS_to_XYZ_mat = np.array([[1.94735469, -1.41445123, 0.36476327],
                                 [0.68990272, 0.34832189, 0.0],
                                 [0.0, 0.0, 1.93485343]])

    @classmethod
    def XYZ_to_LMS(cls, XYZ_values):
        """Transform CIE 1931 XYZ tristimulus values to LMS retinal cone response values."""
        return np.tensordot(XYZ_values, cls._XYZ_to_LMS_mat, axes=(-1, 1))

    @classmethod
    def LMS_to_XYZ(cls, LMS_values):
        """Transform LMS retinal cone response values to CIE 1931 XYZ tristimulus values."""
        return np.tensordot(LMS_values, cls._LMS_to_XYZ_mat, axes=(-1, 1))

    # sRGB D65 XYZ→sRGB matrix (IEC 61966-2-1)
    _CIE1931_XYZ_TO_sRGB_D65 = np.array([[3.2404542, -1.5371385, -0.4985314],
                                          [-0.9692660, 1.8760108, 0.0415560],
                                          [0.0556434, -0.2040259, 1.0572252]])

    @classmethod
    def embed_XYZ_perceived_color_in_sRGB(cls, XYZ_values):
        """Transform CIE 1931 XYZ tristimulus values to sRGB, clipped to [0, 1].

        Parameters
        ----------
        XYZ_values : numpy.ndarray
            XYZ tristimulus values, last axis has length 3.

        Returns
        -------
        numpy.ndarray
            sRGB values in [0, 1], same shape as input.
        """
        tmp = np.tensordot(cls._CIE1931_XYZ_TO_sRGB_D65, XYZ_values, axes=((1,), (-1,)))
        tmp = np.moveaxis(tmp, 0, -1)
        return cls._sRGB_gammacorr(tmp).clip(0., 1.)

    @staticmethod
    def _sRGB_gammacorr(inp):
        """Apply sRGB gamma correction (IEC 61966-2-1)."""
        mask = inp <= 0.0031308
        r1 = 12.92 * inp
        a = 0.055
        r2 = (1 + a) * (np.maximum(inp, 0.0031308) ** (1 / 2.4)) - a
        return np.where(mask, r1, r2)

    # --- CIE standard observer and illuminant data ---

    _CIE1931_STANDARD_OBSERVER_WAVELENGTH_TABLE_380nm_TO_780nm = np.linspace(380., 780., 81)
    _CIE1931_STANDARD_OBSERVER_XYZ_COLOR_MATCHING_TABLE_380nm_TO_780nm = np.array(
          [[0.000160, 0.000662, 0.002362, 0.007242, 0.019110,
            0.043400, 0.084736, 0.140638, 0.204492, 0.264737,
            0.314679, 0.357719, 0.383734, 0.386726, 0.370702,
            0.342957, 0.302273, 0.254085, 0.195618, 0.132349,
            0.080507, 0.041072, 0.016172, 0.005132, 0.003816,
            0.015444, 0.037465, 0.071358, 0.117749, 0.172953,
            0.236491, 0.304213, 0.376772, 0.451584, 0.529826,
            0.616053, 0.705224, 0.793832, 0.878655, 0.951162,
            1.014160, 1.074300, 1.118520, 1.134300, 1.123990,
            1.089100, 1.030480, 0.950740, 0.856297, 0.754930,
            0.647467, 0.535110, 0.431567, 0.343690, 0.268329,
            0.204300, 0.152568, 0.112210, 0.081261, 0.057930,
            0.040851, 0.028623, 0.019941, 0.013842, 0.009577,
            0.006605, 0.004553, 0.003145, 0.002175, 0.001506,
            0.001045, 0.000727, 0.000508, 0.000356, 0.000251,
            0.000178, 0.000126, 0.000090, 0.000065, 0.000046,
            0.000033],
           [0.000017, 0.000072, 0.000253, 0.000769, 0.002004,
            0.004509, 0.008756, 0.014456, 0.021391, 0.029497,
            0.038676, 0.049602, 0.062077, 0.074704, 0.089456,
            0.106256, 0.128201, 0.152761, 0.185190, 0.219940,
            0.253589, 0.297665, 0.339133, 0.395379, 0.460777,
            0.531360, 0.606741, 0.685660, 0.761757, 0.823330,
            0.875211, 0.923810, 0.961988, 0.982200, 0.991761,
            0.999110, 0.997340, 0.982380, 0.955552, 0.915175,
            0.868934, 0.825623, 0.777405, 0.720353, 0.658341,
            0.593878, 0.527963, 0.461834, 0.398057, 0.339554,
            0.283493, 0.228254, 0.179828, 0.140211, 0.107633,
            0.081187, 0.060281, 0.044096, 0.031800, 0.022602,
            0.015905, 0.011130, 0.007749, 0.005375, 0.003718,
            0.002565, 0.001768, 0.001222, 0.000846, 0.000586,
            0.000407, 0.000284, 0.000199, 0.000140, 0.000098,
            0.000070, 0.000050, 0.000036, 0.000025, 0.000018,
            0.000013],
           [0.000705, 0.002928, 0.010482, 0.032344, 0.086011,
            0.197120, 0.389366, 0.656760, 0.972542, 1.282500,
            1.553480, 1.798500, 1.967280, 2.027300, 1.994800,
            1.900700, 1.745370, 1.554900, 1.317560, 1.030200,
            0.772125, 0.570060, 0.415254, 0.302356, 0.218502,
            0.159249, 0.112044, 0.082248, 0.060709, 0.043050,
            0.030451, 0.020584, 0.013676, 0.007918, 0.003988,
            0.001091, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000]])

    @classmethod
    def get_cie1931_standard_observer_XYZ_tristimulus_values(cls, wavelengths):
        """Return CIE 1931 XYZ tristimulus values for given wavelengths (nm).

        Linearly interpolates the 5 nm-spaced standard observer table (380–780 nm).
        Out-of-range wavelengths are clamped to the table endpoints.

        Parameters
        ----------
        wavelengths : numpy.ndarray
            Wavelengths in nm; any shape.

        Returns
        -------
        numpy.ndarray
            Shape ``wavelengths.shape + (3,)``.
        """
        res = np.empty(wavelengths.shape + (3,))
        wl_table = cls._CIE1931_STANDARD_OBSERVER_WAVELENGTH_TABLE_380nm_TO_780nm
        xyz_table = cls._CIE1931_STANDARD_OBSERVER_XYZ_COLOR_MATCHING_TABLE_380nm_TO_780nm
        for i in range(3):
            res[..., i] = np.interp(wavelengths, wl_table, xyz_table[i])
        return res

    # CIE 2022 standard illuminant D65 (DOI: 10.25039/CIE.DS.hjfjmt59)
    _CIE_D65_WAVELENGTH_TABLE_300nm_TO_830nm = np.linspace(300., 830., 531)
    _CIE_D65_RELATIVE_POWER_TABLE_300nm_TO_830nm = np.array(
        [3.41000e-02, 3.60140e-01, 6.86180e-01, 1.01222e+00, 1.33826e+00,
         1.66430e+00, 1.99034e+00, 2.31638e+00, 2.64242e+00, 2.96846e+00,
         3.29450e+00, 4.98865e+00, 6.68280e+00, 8.37695e+00, 1.00711e+01,
         1.17652e+01, 1.34594e+01, 1.51535e+01, 1.68477e+01, 1.85418e+01,
         2.02360e+01, 2.19177e+01, 2.35995e+01, 2.52812e+01, 2.69630e+01,
         2.86447e+01, 3.03265e+01, 3.20082e+01, 3.36900e+01, 3.53717e+01,
         3.70535e+01, 3.73430e+01, 3.76326e+01, 3.79221e+01, 3.82116e+01,
         3.85011e+01, 3.87907e+01, 3.90802e+01, 3.93697e+01, 3.96593e+01,
         3.99488e+01, 4.04451e+01, 4.09414e+01, 4.14377e+01, 4.19340e+01,
         4.24302e+01, 4.29265e+01, 4.34228e+01, 4.39191e+01, 4.44154e+01,
         4.49117e+01, 4.50844e+01, 4.52570e+01, 4.54297e+01, 4.56023e+01,
         4.57750e+01, 4.59477e+01, 4.61203e+01, 4.62930e+01, 4.64656e+01,
         4.66383e+01, 4.71834e+01, 4.77285e+01, 4.82735e+01, 4.88186e+01,
         4.93637e+01, 4.99088e+01, 5.04539e+01, 5.09989e+01, 5.15440e+01,
         5.20891e+01, 5.18777e+01, 5.16664e+01, 5.14550e+01, 5.12437e+01,
         5.10323e+01, 5.08209e+01, 5.06096e+01, 5.03982e+01, 5.01869e+01,
         4.99755e+01, 5.04428e+01, 5.09100e+01, 5.13773e+01, 5.18446e+01,
         5.23118e+01, 5.27791e+01, 5.32464e+01, 5.37137e+01, 5.41809e+01,
         5.46482e+01, 5.74589e+01, 6.02695e+01, 6.30802e+01, 6.58909e+01,
         6.87015e+01, 7.15122e+01, 7.43229e+01, 7.71336e+01, 7.99442e+01,
         8.27549e+01, 8.36280e+01, 8.45011e+01, 8.53742e+01, 8.62473e+01,
         8.71204e+01, 8.79936e+01, 8.88667e+01, 8.97398e+01, 9.06129e+01,
         9.14860e+01, 9.16806e+01, 9.18752e+01, 9.20697e+01, 9.22643e+01,
         9.24589e+01, 9.26535e+01, 9.28481e+01, 9.30426e+01, 9.32372e+01,
         9.34318e+01, 9.27568e+01, 9.20819e+01, 9.14069e+01, 9.07320e+01,
         9.00570e+01, 8.93821e+01, 8.87071e+01, 8.80322e+01, 8.73572e+01,
         8.66823e+01, 8.85006e+01, 9.03188e+01, 9.21371e+01, 9.39554e+01,
         9.57736e+01, 9.75919e+01, 9.94102e+01, 1.01228e+02, 1.03047e+02,
         1.04865e+02, 1.06079e+02, 1.07294e+02, 1.08508e+02, 1.09722e+02,
         1.10936e+02, 1.12151e+02, 1.13365e+02, 1.14579e+02, 1.15794e+02,
         1.17008e+02, 1.17088e+02, 1.17169e+02, 1.17249e+02, 1.17330e+02,
         1.17410e+02, 1.17490e+02, 1.17571e+02, 1.17651e+02, 1.17732e+02,
         1.17812e+02, 1.17517e+02, 1.17222e+02, 1.16927e+02, 1.16632e+02,
         1.16336e+02, 1.16041e+02, 1.15746e+02, 1.15451e+02, 1.15156e+02,
         1.14861e+02, 1.14967e+02, 1.15073e+02, 1.15180e+02, 1.15286e+02,
         1.15392e+02, 1.15498e+02, 1.15604e+02, 1.15711e+02, 1.15817e+02,
         1.15923e+02, 1.15212e+02, 1.14501e+02, 1.13789e+02, 1.13078e+02,
         1.12367e+02, 1.11656e+02, 1.10945e+02, 1.10233e+02, 1.09522e+02,
         1.08811e+02, 1.08865e+02, 1.08920e+02, 1.08974e+02, 1.09028e+02,
         1.09082e+02, 1.09137e+02, 1.09191e+02, 1.09245e+02, 1.09300e+02,
         1.09354e+02, 1.09199e+02, 1.09044e+02, 1.08888e+02, 1.08733e+02,
         1.08578e+02, 1.08423e+02, 1.08268e+02, 1.08112e+02, 1.07957e+02,
         1.07802e+02, 1.07501e+02, 1.07200e+02, 1.06898e+02, 1.06597e+02,
         1.06296e+02, 1.05995e+02, 1.05694e+02, 1.05392e+02, 1.05091e+02,
         1.04790e+02, 1.05080e+02, 1.05370e+02, 1.05660e+02, 1.05950e+02,
         1.06239e+02, 1.06529e+02, 1.06819e+02, 1.07109e+02, 1.07399e+02,
         1.07689e+02, 1.07361e+02, 1.07032e+02, 1.06704e+02, 1.06375e+02,
         1.06047e+02, 1.05719e+02, 1.05390e+02, 1.05062e+02, 1.04733e+02,
         1.04405e+02, 1.04369e+02, 1.04333e+02, 1.04297e+02, 1.04261e+02,
         1.04225e+02, 1.04190e+02, 1.04154e+02, 1.04118e+02, 1.04082e+02,
         1.04046e+02, 1.03641e+02, 1.03237e+02, 1.02832e+02, 1.02428e+02,
         1.02023e+02, 1.01618e+02, 1.01214e+02, 1.00809e+02, 1.00405e+02,
         1.00000e+02, 9.96334e+01, 9.92668e+01, 9.89003e+01, 9.85337e+01,
         9.81671e+01, 9.78005e+01, 9.74339e+01, 9.70674e+01, 9.67008e+01,
         9.63342e+01, 9.62796e+01, 9.62250e+01, 9.61703e+01, 9.61157e+01,
         9.60611e+01, 9.60065e+01, 9.59519e+01, 9.58972e+01, 9.58426e+01,
         9.57880e+01, 9.50778e+01, 9.43675e+01, 9.36573e+01, 9.29470e+01,
         9.22368e+01, 9.15266e+01, 9.08163e+01, 9.01061e+01, 8.93958e+01,
         8.86856e+01, 8.88177e+01, 8.89497e+01, 8.90818e+01, 8.92138e+01,
         8.93459e+01, 8.94780e+01, 8.96100e+01, 8.97421e+01, 8.98741e+01,
         9.00062e+01, 8.99655e+01, 8.99248e+01, 8.98841e+01, 8.98434e+01,
         8.98026e+01, 8.97619e+01, 8.97212e+01, 8.96805e+01, 8.96398e+01,
         8.95991e+01, 8.94091e+01, 8.92190e+01, 8.90290e+01, 8.88389e+01,
         8.86489e+01, 8.84589e+01, 8.82688e+01, 8.80788e+01, 8.78887e+01,
         8.76987e+01, 8.72577e+01, 8.68167e+01, 8.63757e+01, 8.59347e+01,
         8.54936e+01, 8.50526e+01, 8.46116e+01, 8.41706e+01, 8.37296e+01,
         8.32886e+01, 8.33297e+01, 8.33707e+01, 8.34118e+01, 8.34528e+01,
         8.34939e+01, 8.35350e+01, 8.35760e+01, 8.36171e+01, 8.36581e+01,
         8.36992e+01, 8.33320e+01, 8.29647e+01, 8.25975e+01, 8.22302e+01,
         8.18630e+01, 8.14958e+01, 8.11285e+01, 8.07613e+01, 8.03940e+01,
         8.00268e+01, 8.00456e+01, 8.00644e+01, 8.00831e+01, 8.01019e+01,
         8.01207e+01, 8.01395e+01, 8.01583e+01, 8.01770e+01, 8.01958e+01,
         8.02146e+01, 8.04209e+01, 8.06272e+01, 8.08336e+01, 8.10399e+01,
         8.12462e+01, 8.14525e+01, 8.16588e+01, 8.18652e+01, 8.20715e+01,
         8.22778e+01, 8.18784e+01, 8.14791e+01, 8.10797e+01, 8.06804e+01,
         8.02810e+01, 7.98816e+01, 7.94823e+01, 7.90829e+01, 7.86836e+01,
         7.82842e+01, 7.74279e+01, 7.65716e+01, 7.57153e+01, 7.48590e+01,
         7.40027e+01, 7.31465e+01, 7.22902e+01, 7.14339e+01, 7.05776e+01,
         6.97213e+01, 6.99101e+01, 7.00989e+01, 7.02876e+01, 7.04764e+01,
         7.06652e+01, 7.08540e+01, 7.10428e+01, 7.12315e+01, 7.14203e+01,
         7.16091e+01, 7.18831e+01, 7.21571e+01, 7.24311e+01, 7.27051e+01,
         7.29790e+01, 7.32530e+01, 7.35270e+01, 7.38010e+01, 7.40750e+01,
         7.43490e+01, 7.30745e+01, 7.18000e+01, 7.05255e+01, 6.92510e+01,
         6.79765e+01, 6.67020e+01, 6.54275e+01, 6.41530e+01, 6.28785e+01,
         6.16040e+01, 6.24322e+01, 6.32603e+01, 6.40885e+01, 6.49166e+01,
         6.57448e+01, 6.65730e+01, 6.74011e+01, 6.82293e+01, 6.90574e+01,
         6.98856e+01, 7.04057e+01, 7.09259e+01, 7.14460e+01, 7.19662e+01,
         7.24863e+01, 7.30064e+01, 7.35266e+01, 7.40467e+01, 7.45669e+01,
         7.50870e+01, 7.39376e+01, 7.27881e+01, 7.16387e+01, 7.04893e+01,
         6.93398e+01, 6.81904e+01, 6.70410e+01, 6.58916e+01, 6.47421e+01,
         6.35927e+01, 6.18752e+01, 6.01578e+01, 5.84403e+01, 5.67229e+01,
         5.50054e+01, 5.32880e+01, 5.15705e+01, 4.98531e+01, 4.81356e+01,
         4.64182e+01, 4.84569e+01, 5.04956e+01, 5.25344e+01, 5.45731e+01,
         5.66118e+01, 5.86505e+01, 6.06892e+01, 6.27280e+01, 6.47667e+01,
         6.68054e+01, 6.64631e+01, 6.61209e+01, 6.57786e+01, 6.54364e+01,
         6.50941e+01, 6.47518e+01, 6.44096e+01, 6.40673e+01, 6.37251e+01,
         6.33828e+01, 6.34749e+01, 6.35670e+01, 6.36592e+01, 6.37513e+01,
         6.38434e+01, 6.39355e+01, 6.40276e+01, 6.41198e+01, 6.42119e+01,
         6.43040e+01, 6.38188e+01, 6.33336e+01, 6.28484e+01, 6.23632e+01,
         6.18779e+01, 6.13927e+01, 6.09075e+01, 6.04223e+01, 5.99371e+01,
         5.94519e+01, 5.87026e+01, 5.79533e+01, 5.72040e+01, 5.64547e+01,
         5.57054e+01, 5.49562e+01, 5.42069e+01, 5.34576e+01, 5.27083e+01,
         5.19590e+01, 5.25072e+01, 5.30553e+01, 5.36035e+01, 5.41516e+01,
         5.46998e+01, 5.52480e+01, 5.57961e+01, 5.63443e+01, 5.68924e+01,
         5.74406e+01, 5.77278e+01, 5.80150e+01, 5.83022e+01, 5.85894e+01,
         5.88765e+01, 5.91637e+01, 5.94509e+01, 5.97381e+01, 6.00253e+01,
         6.03125e+01])

    @classmethod
    def get_cie_d65_standard_illuminant_power(cls, wavelengths):
        """Return relative spectral power of the CIE D65 illuminant at given wavelengths (nm).

        Parameters
        ----------
        wavelengths : numpy.ndarray
            Wavelengths in nm; any shape.

        Returns
        -------
        numpy.ndarray
            Relative power values, same shape as input.
        """
        return np.interp(wavelengths,
                         cls._CIE_D65_WAVELENGTH_TABLE_300nm_TO_830nm,
                         cls._CIE_D65_RELATIVE_POWER_TABLE_300nm_TO_830nm)

    # --- Post-processing ---

    @staticmethod
    def enhance_sRGB_color_contrast(sRGB_data, color_contrast_multiplier):
        """Enhance color saturation of an sRGB image while preserving black and white points.

        Parameters
        ----------
        sRGB_data : numpy.ndarray
            sRGB values in [0, 1].
        color_contrast_multiplier : float
            Saturation scale factor. 1.0 returns the input unchanged.

        Returns
        -------
        numpy.ndarray
            Contrast-enhanced sRGB values in [0, 1].
        """
        if color_contrast_multiplier == 1.0:
            return sRGB_data

        if np.any(sRGB_data < 0.0) or np.any(sRGB_data > 1.0):
            raise ValueError("sRGB data must be in [0, 1]")

        black_mask = np.all(sRGB_data[..., :3] == 0., axis=-1)
        white_mask = np.all(sRGB_data[..., :3] == 1., axis=-1)

        grey_vals = (0.2989 * sRGB_data[..., 0]
                     + 0.5870 * sRGB_data[..., 1]
                     + 0.1140 * sRGB_data[..., 2])[..., np.newaxis]

        res = sRGB_data.copy()
        res[..., :3] = (color_contrast_multiplier * (sRGB_data[..., :3] - grey_vals)
                        + grey_vals).clip(0., 1.)
        res[black_mask, :3] = 0.
        res[white_mask, :3] = 1.
        return res


class SpectrumToRGBProjector:
    """Project fields with a spectral dimension to sRGB color space.

    The spectral dimension is mapped onto the visible light spectral range, converted
    to perceived colors following the CIE 1931 model of human color perception, and
    encoded into sRGB.

    The class distinguishes between total spectral bin flux :math:`\\Phi_k` and spectral
    flux density :math:`\\partial\\Phi/\\partial \\left[E \\mid \\lambda \\right]`,
    providing separate projection methods for each:
    :meth:`project_total_spectral_bin_flux` and :meth:`project_spectral_flux_density`.

    **Setup** (must be called before projecting):

    1. Specify input bin boundaries via
       :meth:`specify_input_spectrum_bins_via_bin_boundaries` or
       :meth:`specify_input_spectrum_bins_via_center_and_width`.
    2. Optionally fix a white point via :meth:`set_saturation_flux` or
       :meth:`set_saturation_flux_density`.

    Parameters
    ----------
    spectral_axis_type : {'energy', 'wavelength'}
        Physical meaning of the input spectral coordinate; controls mapping
        *direction* only.

        - ``'energy'`` *(default)*: direction-inverting — lower input maps to
          the red end and higher input maps to the blue end, consistent with
          E = hc/λ.  Suited to energy- or frequency-domain data.
        - ``'wavelength'``: direction-preserving — smaller input values map to
          the blue end and larger values to the red end.  Pass actual wavelength
          values as the ``centers`` / boundary arguments.
    visible_bin_width : {'uniform', 'proportional'} or None
        How the visible wavelength width is distributed across input bins.

        - ``'uniform'``: every input bin receives the same visible width
          Δλ = (λ_max − λ_min) / N, regardless of its width in the input
          domain.  Ideal for energy-domain data (especially geomspace bins)
          where equal visual treatment of each bin is desired.
        - ``'proportional'``: each bin's visible width is proportional to its
          width in the input domain, preserving relative spectral coverage.
          Ideal for wavelength-domain data.
    wavelength_min_mappable : float
        Short-wavelength limit (nm) of the visible range to map onto.
        The highest input bin maps to this wavelength. Default: 440 nm.
    wavelength_max_mappable : float
        Long-wavelength limit (nm) of the visible range to map onto.
        The lowest input bin maps to this wavelength. Default: 640 nm.
    """

    _input_spectrum_bin_lower = None
    _input_spectrum_bin_upper = None
    _input_spectrum_bin_widths = None
    _input_spectrum_relative_bin_widths = None

    _spectral_axis_type = None
    _visible_bin_width = None
    _input_saturation_flux = None

    _visible_spectrum_bin_lower_wavelengths = None
    _visible_spectrum_bin_upper_wavelengths = None
    _visible_spectrum_bin_widths = None

    _visible_spectrum_bin_flux_to_XYZ_mapping_tensor = None

    def __init__(self, spectral_axis_type, visible_bin_width,
                 wavelength_min_mappable=440., wavelength_max_mappable=640.):
        if spectral_axis_type not in ('energy', 'wavelength'):
            raise ValueError("spectral_axis_type must be 'energy' or 'wavelength'")
        self._spectral_axis_type = spectral_axis_type
        if visible_bin_width not in ('uniform', 'proportional'):
            raise ValueError("visible_bin_width must be 'uniform' or 'proportional'")
        self._visible_bin_width = visible_bin_width
        self._WAVELENGTH_MIN_MAPPABLE = self._check_pos_scalar(
            wavelength_min_mappable, "wavelength_min_mappable")
        self._WAVELENGTH_MAX_MAPPABLE = self._check_pos_scalar(
            wavelength_max_mappable, "wavelength_max_mappable")

    # --- Bin specification ---

    def specify_input_spectrum_bins_via_bin_boundaries(self, lower, upper):
        """Specify input spectral bins by their lower and upper boundaries.

        Parameters
        ----------
        lower : array-like
            Lower boundary of each bin.
        upper : array-like
            Upper boundary of each bin (must match length of lower).
        """
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        if len(lower) != len(upper):
            raise ValueError("lower and upper must have the same length")
        widths = upper - lower
        if np.any(widths <= 0.):
            raise ValueError("all bin widths must be positive")
        sort_idx_lower = np.argsort(lower)
        sort_idx_upper = np.argsort(upper)
        if not np.all(sort_idx_lower == sort_idx_upper):
            raise ValueError("bins must be sorted consistently and must not overlap")
        if np.any(upper[sort_idx_upper][:-1] > lower[sort_idx_lower][1:]):
            raise ValueError("spectral bins cannot overlap")

        self._input_spectrum_bin_lower = lower
        self._input_spectrum_bin_upper = upper

        if self._input_spectrum_bin_widths is None:
            self._input_spectrum_bin_widths = widths
            self._input_spectrum_relative_bin_widths = widths / np.sum(widths)
        else:
            if not np.allclose(widths, self._input_spectrum_bin_widths):
                raise ValueError("inconsistent bin definition — did you call multiple "
                                 "bin specification methods?")

    def specify_input_spectrum_bins_via_center_and_width(self, centers, widths):
        """Specify input bins by their center coordinates and widths.

        Parameters
        ----------
        centers : array-like
            Center coordinate of each bin.
        widths : array-like
            Width of each bin (center ± width/2).
        """
        centers = np.asarray(centers, dtype=float)
        widths = np.asarray(widths, dtype=float)
        if np.any(widths <= 0.):
            raise ValueError("bin widths must be positive")
        self._input_spectrum_bin_widths = widths
        self._input_spectrum_relative_bin_widths = widths / np.sum(widths)
        self.specify_input_spectrum_bins_via_bin_boundaries(
            centers - widths / 2., centers + widths / 2.)

    # --- White-point specification ---

    def set_saturation_flux(self, saturation_flux):
        """Fix the white point as a total (spectrally integrated) flux value.

        Parameters
        ----------
        saturation_flux : float
            Total flux that defines the upper saturation (white) point.
        """
        self._input_saturation_flux = float(
            self._check_pos_scalar(saturation_flux, "saturation_flux"))

    def set_saturation_flux_density(self, saturation_flux_density, spectral_denseness):
        """Fix the white point via a flux density and an estimate of spectral filling.

        Because brightness perception scales with total flux, not flux density, we need
        to know over what spectral range to integrate. ``spectral_denseness`` encodes
        this: use ~1.0 for flat spectra and ~1/(peak width) for strongly peaked ones.

        Requires that bin widths have already been specified.

        Parameters
        ----------
        saturation_flux_density : float
            Flux density defining the upper saturation point.
        spectral_denseness : float in (0, 1]
            Fraction of the spectral domain that is effectively filled by the source.
        """
        saturation_flux_density = float(
            self._check_pos_scalar(saturation_flux_density, "saturation_flux_density"))
        if not self._is_scalar(spectral_denseness):
            raise ValueError("spectral_denseness must be a scalar")
        if not (0. < spectral_denseness <= 1.):
            raise ValueError("spectral_denseness must be in (0, 1]")
        if self._input_spectrum_bin_widths is None:
            raise RuntimeError("specify bin widths before calling set_saturation_flux_density")
        self._input_saturation_flux = (saturation_flux_density
                                       * spectral_denseness
                                       * np.sum(self._input_spectrum_bin_widths))

    # --- Projection ---

    def project_spectral_flux_density(self, spectral_flux_density, saturation_via='luminance',
                                      dynamic_range=None, XYZ_inspect_callback=None):
        """Project spectral flux density data to sRGB perceived colors.

        Multiplies each bin's density value by its width to obtain total bin flux,
        then delegates to :meth:`project_total_spectral_bin_flux`.

        Parameters
        ----------
        spectral_flux_density : numpy.ndarray
            Flux density values; spectral axis must be last.
        saturation_via : {'luminance', 'retinal cone response'}
            Saturation model. Default: 'luminance'.
        dynamic_range : float or None
            If set, applies log tone-mapping on luminance after saturation,
            compressing the range [saturation/dynamic_range, saturation] to [0, 1].
            None (default) uses linear saturation only.
        XYZ_inspect_callback : callable or None
            Called as ``fn(XYZ_raw, XYZ_saturated)`` for inspection. Default: None.

        Returns
        -------
        numpy.ndarray
            sRGB values in [0, 1]; last axis has length 3.
        """
        self._pre_projection_checks(spectral_flux_density)
        broadcast_sl = (None,) * (spectral_flux_density.ndim - 1) + (slice(None),)
        total_spectral_bin_flux = (spectral_flux_density
                                   * self._input_spectrum_bin_widths[broadcast_sl])
        return self.project_total_spectral_bin_flux(
            total_spectral_bin_flux,
            saturation_via=saturation_via,
            dynamic_range=dynamic_range,
            XYZ_inspect_callback=XYZ_inspect_callback)

    def project_total_spectral_bin_flux(self, total_spectral_bin_flux, saturation_via='luminance',
                                        dynamic_range=None, XYZ_inspect_callback=None):
        """Project total spectral bin flux data to sRGB perceived colors.

        Parameters
        ----------
        total_spectral_bin_flux : numpy.ndarray
            Total flux per bin; spectral axis must be last.
        saturation_via : {'luminance', 'retinal cone response'}
            Saturation model. Default: 'luminance'.
        dynamic_range : float or None
            If set, applies log tone-mapping on luminance after saturation,
            compressing the range [saturation/dynamic_range, saturation] to [0, 1].
            None (default) uses linear saturation only.
        XYZ_inspect_callback : callable or None
            Called as ``fn(XYZ_raw, XYZ_saturated)`` for inspection. Default: None.

        Returns
        -------
        numpy.ndarray
            sRGB values in [0, 1]; last axis has length 3.
        """
        self._pre_projection_checks(total_spectral_bin_flux)

        XYZ_data = self._transform_visible_spectrum_bin_flux_to_XYZ(total_spectral_bin_flux)

        saturation_reference = None if self._input_saturation_flux is None else \
            self._input_saturation_flux * self._input_spectrum_relative_bin_widths

        if saturation_via == 'luminance':
            saturation_fn = self._apply_luminance_saturation
        elif saturation_via == 'retinal cone response':
            saturation_fn = self._apply_cone_response_saturation
        else:
            raise ValueError(f"Unknown saturation mode '{saturation_via}'; "
                             "expected 'luminance' or 'retinal cone response'")

        XYZ_saturated = saturation_fn(XYZ_data, saturation_reference)

        if dynamic_range is not None:
            XYZ_saturated = self._apply_log_tone_mapping(XYZ_saturated, dynamic_range)

        if XYZ_inspect_callback is not None:
            if not callable(XYZ_inspect_callback):
                raise TypeError("XYZ_inspect_callback must be callable")
            XYZ_inspect_callback(XYZ_data, XYZ_saturated)

        return ColorSpaceTools.embed_XYZ_perceived_color_in_sRGB(XYZ_saturated)

    # --- Saturation implementations ---

    def _apply_luminance_saturation(self, XYZ_data, reference_bin_flux_spectrum=None):
        """Normalise XYZ by a reference luminance so that Y ∈ [0, 1].

        Parameters
        ----------
        XYZ_data : numpy.ndarray
            Raw XYZ values to normalise.
        reference_bin_flux_spectrum : numpy.ndarray or None
            If given, derive the saturation luminance from the Y value this
            spectrum produces. Otherwise use the maximum Y in XYZ_data.
        """
        if reference_bin_flux_spectrum is None:
            saturation_luminance = np.max(XYZ_data[..., 1])
        else:
            XYZ_reference = self._transform_visible_spectrum_bin_flux_to_XYZ(
                reference_bin_flux_spectrum)
            saturation_luminance = XYZ_reference[1]
        if saturation_luminance == 0.:
            return XYZ_data  # all-zero input, nothing to normalise
        return XYZ_data / saturation_luminance

    def _apply_cone_response_saturation(self, XYZ_data, reference_bin_flux_spectrum=None):
        """Normalise in LMS (retinal cone) space, then convert back to XYZ.

        Parameters
        ----------
        XYZ_data : numpy.ndarray
            Raw XYZ values to normalise.
        reference_bin_flux_spectrum : numpy.ndarray or None
            If given, derive the per-channel LMS saturation from this spectrum.
            Otherwise use the maximum per-channel LMS value in LMS_data.
        """
        LMS_data = ColorSpaceTools.XYZ_to_LMS(XYZ_data)

        if reference_bin_flux_spectrum is None:
            saturation_LMS = np.array([np.max(LMS_data[..., i]) for i in range(3)])
        else:
            ref = reference_bin_flux_spectrum[np.newaxis, :]
            reference_XYZ = self._transform_visible_spectrum_bin_flux_to_XYZ(ref)
            saturation_LMS = ColorSpaceTools.XYZ_to_LMS(reference_XYZ)[0]

        broadcast_sl = (None,) * (LMS_data.ndim - 1) + (slice(None),)
        LMS_data = LMS_data / saturation_LMS[broadcast_sl]
        return ColorSpaceTools.LMS_to_XYZ(LMS_data.clip(0., 1.))

    # --- Tone mapping ---

    def _apply_log_tone_mapping(self, XYZ_data, dynamic_range):
        """Apply log-scale tone mapping to the luminance channel.

        Scales all XYZ channels by log(Y)/Y, preserving chromaticity while
        compressing the luminance from a linear to a log scale.

        Parameters
        ----------
        XYZ_data : numpy.ndarray
            XYZ values with Y expected in [0, 1] after saturation normalisation.
        dynamic_range : float
            Ratio of the white-point luminance to the black-point luminance.
            Pixels below Y = 1/dynamic_range are rendered black.

        Returns
        -------
        numpy.ndarray
            Tone-mapped XYZ values.
        """
        Y = XYZ_data[..., 1:2]  # shape (..., 1) for broadcasting across X, Y, Z
        Y_log = self._to_logscale(Y, 1. / dynamic_range, 1.)
        scale = np.where(Y > 0, Y_log / np.where(Y > 0, Y, 1.), 0.)
        return XYZ_data * scale

    # --- Visible-spectrum bin mapping ---

    def _map_input_spectrum_bins_to_visible_light_wavelength_bins(self):
        """Map input spectral bin boundaries onto the visible wavelength range.

        Two constructor arguments control the mapping independently:

        - ``visible_bin_width``: ``'uniform'`` gives every bin equal Δλ;
          ``'proportional'`` makes each bin's visible width proportional to
          its input-domain width.
        - ``spectral_axis_type``: ``'energy'`` is direction-inverting (lower input →
          red end); ``'wavelength'`` is direction-preserving (lower input →
          blue end).

        The return arrays always satisfy ``lam_lower[k] < lam_upper[k]``
        (shorter wavelength first).

        Returns
        -------
        lam_lower, lam_upper : numpy.ndarray
            Visible wavelength boundaries (nm) for each input bin,
            with ``lam_lower[k] < lam_upper[k]`` for all k.
        """
        in_lower = self._input_spectrum_bin_lower
        in_upper = self._input_spectrum_bin_upper

        if np.any(in_upper - in_lower <= 0):
            raise ValueError("bins of zero or negative width given")

        lam_min = self._WAVELENGTH_MIN_MAPPABLE
        lam_max = self._WAVELENGTH_MAX_MAPPABLE
        delta = lam_max - lam_min

        # Step 1: normalised positions t ∈ [0, 1] per bin boundary
        if self._visible_bin_width == 'uniform':
            N = len(in_lower)
            k = np.arange(N)
            t_lower, t_upper = k / N, (k + 1) / N
        else:  # 'proportional'
            in_min = np.min(in_lower)
            in_max = np.max(in_upper)
            t_lower = (in_lower - in_min) / (in_max - in_min)
            t_upper = (in_upper - in_min) / (in_max - in_min)

        # Step 2: map t to visible wavelengths according to direction
        if self._spectral_axis_type == 'wavelength':
            # direction-preserving: t=0 → blue end (lam_min), t=1 → red end (lam_max)
            lam_lower = lam_min + t_lower * delta
            lam_upper = lam_min + t_upper * delta
        else:  # 'energy'
            # direction-inverting: t=0 → red end (lam_max), t=1 → blue end (lam_min)
            # swap t_lower/t_upper so that lam_lower < lam_upper is preserved
            lam_lower = lam_min + (1. - t_upper) * delta
            lam_upper = lam_min + (1. - t_lower) * delta

        return lam_lower, lam_upper

    def _generate_visible_spectrum_bin_flux_to_XYZ_mapping_tensor(self):
        """Pre-compute the (n_bins, 3) mapping from visible bin flux to XYZ.

        For each visible bin, averages the CIE 1931 tristimulus values over 100
        wavelength samples drawn uniformly in wavelength within the bin, matching
        the CIE definition XYZ = ∫ P(λ) CMF(λ) dλ for emissive sources.
        """
        lam_lower, lam_upper = self._map_input_spectrum_bins_to_visible_light_wavelength_bins()
        self._visible_spectrum_bin_lower_wavelengths = lam_lower
        self._visible_spectrum_bin_upper_wavelengths = lam_upper
        self._visible_spectrum_bin_widths = lam_upper - lam_lower

        # Sample 100 wavelengths per bin, uniformly spaced in wavelength
        within_bin_wavelengths = np.array(
            [np.linspace(wl_lo, wl_hi, 100)
             for wl_lo, wl_hi in zip(lam_lower, lam_upper)])

        XYZ_samples = ColorSpaceTools.get_cie1931_standard_observer_XYZ_tristimulus_values(
            within_bin_wavelengths)
        self._visible_spectrum_bin_flux_to_XYZ_mapping_tensor = np.mean(XYZ_samples, axis=1)

    def _transform_visible_spectrum_bin_flux_to_XYZ(self, vis_bin_flux):
        return np.tensordot(vis_bin_flux,
                            self._visible_spectrum_bin_flux_to_XYZ_mapping_tensor,
                            axes=(-1, 0))

    # --- Pre-flight checks ---

    def _pre_projection_checks(self, input_data):
        if self._input_spectrum_bin_widths is None:
            raise RuntimeError("specify spectral bins before projecting")
        if np.any(input_data < 0.):
            raise ValueError("fluxes and flux densities must be non-negative")
        if input_data.shape[-1] != self._input_spectrum_bin_lower.shape[0]:
            raise ValueError("last dimension of input does not match number of spectral bins")
        if self._visible_spectrum_bin_flux_to_XYZ_mapping_tensor is None:
            self._generate_visible_spectrum_bin_flux_to_XYZ_mapping_tensor()

    # --- Utilities ---

    def _to_logscale(self, arr, vmin, vmax):
        res = arr.clip(vmin, vmax)
        res = np.log(res / vmin)
        res /= np.log(vmax / vmin)
        return res

    def get_color_map_image(self, total_flux_levels=None, flux_density_levels=None,
                            saturation_via='luminance'):
        """Return an sRGB image showing the color assigned to each spectral bin.

        Produces a 2-D image of shape ``(n_levels, n_bins, 3)`` where each row
        corresponds to one flux level and each column to one input bin with a
        monochromatic (one-hot) spectrum at that level.  All other bins are zero.

        Exactly one of ``total_flux_levels`` or ``flux_density_levels`` must be
        provided.

        Parameters
        ----------
        total_flux_levels : numpy.ndarray, 1-D or None
            Total bin flux values at which to sample the color map.
            Uses :meth:`project_total_spectral_bin_flux`.
        flux_density_levels : numpy.ndarray, 1-D or None
            Spectral flux density values at which to sample the color map.
            Uses :meth:`project_spectral_flux_density`.
        saturation_via : str
            Saturation model passed to the projection method. Default: 'luminance'.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_levels, n_bins, 3)`` of sRGB values in [0, 1].
        """
        if (total_flux_levels is None) == (flux_density_levels is None):
            raise ValueError(
                "provide exactly one of total_flux_levels or flux_density_levels")

        if total_flux_levels is not None:
            levels = np.asarray(total_flux_levels)
            project = self.project_total_spectral_bin_flux
        else:
            levels = np.asarray(flux_density_levels)
            project = self.project_spectral_flux_density

        if levels.ndim != 1:
            raise ValueError("flux levels must be a 1-D array")

        n_bins = len(self._input_spectrum_bin_lower)
        probe = np.eye(n_bins)[np.newaxis, :, :] * levels[:, np.newaxis, np.newaxis]
        return project(probe, saturation_via=saturation_via)

    # --- Scalar validation helpers ---

    def _check_pos_scalar(self, inp, name):
        if not self._is_scalar(inp):
            raise ValueError(f"{name} must be a scalar")
        if inp <= 0.:
            raise ValueError(f"{name} must be strictly positive")
        return inp

    def _is_scalar(self, inp):
        import numbers
        return isinstance(inp, numbers.Number) or (
            isinstance(inp, np.ndarray) and inp.shape == ())
