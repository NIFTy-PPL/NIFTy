# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2013-2026 Max-Planck-Society
# Copyright(C) 2026 Lukas Scheel-Platz
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from .logger import logger


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

    **Setup** (must be called before projecting):

    1. Specify input bin boundaries via
       :meth:`specify_input_spectrum_bins_via_bin_boundaries` or
       :meth:`specify_input_spectrum_bins_via_center_and_width`.
    2. Optionally fix the displayed luminance range via :meth:`set_luminance_range`, and
       optionally enable log compression via :meth:`use_log_compression`.

    The display transform is fully described by two luminance values
    :math:`(Y_\\mathrm{black}, Y_\\mathrm{saturation})`, a tone curve (linear or logarithmic)
    and a highlight policy.  Physical quantities are turned into luminances by the
    converter methods :meth:`luminance_of_spectrum` and :meth:`luminance_quantiles`,
    whose results are handed to :meth:`set_luminance_range`.

    The input spectral range is mapped onto the configured window
    ``[wavelength_min_mappable, wavelength_max_mappable]``, which is a *display*
    choice and carries no physical wavelength meaning.  Within that window the
    placement is affine — in the input coordinate for
    ``visible_bin_width='proportional'``, in the bin index for ``'uniform'`` — so
    the assigned wavelengths are **not** related to the input by any physical law.
    ``spectral_axis_type`` fixes only which end of the window the low-input bins
    land on.

    Parameters
    ----------
    flux_convention : {'bin_integrated_flux', 'flux_density'}
        Physical convention of *all* data passed to this projector — both the images
        handed to :meth:`project` and the spectra handed to the luminance converters.

        - ``'bin_integrated_flux'``: values are total fluxes :math:`\\Phi_k`, already
          integrated over their spectral bin.
        - ``'flux_density'``: values are densities
          :math:`\\partial\\Phi/\\partial\\left[E \\mid \\lambda\\right]`, which get
          multiplied by the bin widths internally.

        The chosen convention is logged at construction time, because silently
        reusing a projector on data of the other kind produces a plausible-looking
        but wrong image.
    spectral_axis_type : {'energy', 'wavelength'}
        Physical meaning of the input spectral coordinate; controls the mapping
        *direction* only, never its spacing.

        - ``'energy'``: direction-inverting — the lowest input bin is placed at
          ``wavelength_max_mappable`` (red end) and the highest at
          ``wavelength_min_mappable`` (blue end).  Only the *ordering* agrees with
          E = hc/λ; the spacing does not, since λ decreases affinely in E rather
          than as 1/E.  Suited to energy- or frequency-domain data.
        - ``'wavelength'``: direction-preserving — the lowest input bin is placed
          at ``wavelength_min_mappable`` (blue end) and the highest at
          ``wavelength_max_mappable`` (red end).  Note that the input wavelengths
          are still rescaled onto the configured window, not used as they are.
    visible_bin_width : {'uniform', 'proportional'}
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

    _FLUX_CONVENTIONS = ('bin_integrated_flux', 'flux_density')
    _HIGHLIGHT_MODES = ('clamp', 'clip_channels')

    def __init__(self, flux_convention, spectral_axis_type, visible_bin_width,
                 wavelength_min_mappable=440., wavelength_max_mappable=640.):
        if flux_convention not in self._FLUX_CONVENTIONS:
            raise ValueError("flux_convention must be one of "
                             f"{self._FLUX_CONVENTIONS}")
        self._flux_convention = flux_convention
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

        # --- input spectrum bins ---
        self._input_spectrum_bin_lower = None
        self._input_spectrum_bin_upper = None
        self._input_spectrum_bin_widths = None
        self._input_spectrum_relative_bin_widths = None

        # --- visible spectrum bins ---
        self._visible_spectrum_bin_lower_wavelengths = None
        self._visible_spectrum_bin_upper_wavelengths = None
        self._visible_spectrum_bin_widths = None
        self._visible_spectrum_bin_flux_to_XYZ_mapping_tensor = None

        # --- display transform ---
        self._Y_black = None
        self._Y_saturation = None
        self._highlights = 'clamp'
        self._log_compression = False

        # --- state recorded by the last projection ---
        self._last_transform = None   # (Y_black, Y_saturation, log, highlights)
        self._last_flux_range = None  # (min, min_positive, max)
        self._last_color_map_levels = None
        self._auto_saturation_warned = False

        logger.info("SpectrumToRGBProjector: interpreting input as %s",
                    "bin-integrated flux" if flux_convention == 'bin_integrated_flux'
                    else "spectral flux density")

    # --- Bin specification ---

    def specify_input_spectrum_bins_via_bin_boundaries(self, lower, upper):
        """Specify input spectral bins by their lower and upper boundaries.

        Discards any luminance range, tone curve state and cached colour map, since
        those are only meaningful relative to a fixed set of bins.

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
        self._invalidate_derived_state()

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

    def _invalidate_derived_state(self):
        self._visible_spectrum_bin_flux_to_XYZ_mapping_tensor = None
        self._Y_black = None
        self._Y_saturation = None
        self._last_transform = None
        self._last_flux_range = None
        self._last_color_map_levels = None

    # --- Luminance converters (pure: no state is modified) ---

    def luminance_of_spectrum(self, spectrum):
        """Return the luminance Y that a given spectrum is rendered with.

        The spectrum is interpreted according to this projector's
        ``flux_convention``.  This is the intended way to turn a physical
        specification into the luminance values understood by
        :meth:`set_luminance_range`, e.g.::

            # saturation luminance: a flat spectrum carrying a total flux of F
            widths = ...   # the bin widths handed to specify_input_spectrum_bins_*
            flat = F*widths/widths.sum()    # 'bin_integrated_flux' convention
            flat = np.full(n_bins, F/widths.sum())      # 'flux_density' convention
            proj.set_luminance_range(Y_saturation=proj.luminance_of_spectrum(flat))

        Parameters
        ----------
        spectrum : array-like
            Spectrum with the spectral axis last; leading axes are allowed.

        Returns
        -------
        float or numpy.ndarray
            Luminance of each spectrum.
        """
        spectrum = np.asarray(spectrum, dtype=float)
        self._check_spectral_shape(spectrum)
        self._ensure_mapping_tensor()
        return self._transform_input_flux_to_XYZ(spectrum)[..., 1]

    def luminance_quantiles(self, data, q=(0.01, 0.99)):
        """Return luminance quantiles of a sample spatio-spectral image.

        Every pixel of ``data`` is converted to a luminance, and the requested
        quantiles of the resulting distribution are returned.  This is the natural
        way to pick a display range: it asks the data where its interesting
        luminances lie rather than requiring the caller to guess absolute values.

        Non-positive luminances (zero or negative) are excluded before quantiling.
        Masked or zero-padded fields are frequently majority-zero, which would
        otherwise pin the lower quantile to exactly zero and make log compression
        fail far away from its cause.

        Parameters
        ----------
        data : array-like
            Spatio-spectral data with the spectral axis last, interpreted according
            to this projector's ``flux_convention``.
        q : pair of float
            Lower and upper quantile in [0, 1]. Default: (0.01, 0.99).

            Note that log rendering usually wants a considerably higher lower
            quantile (~0.3-0.5, near the background level of the image) than the
            default, which is chosen to be unsurprising for linear rendering.

        Returns
        -------
        (float, float)
            Luminance at the lower and upper quantile.
        """
        q_low, q_high = (float(qq) for qq in q)
        if not (0. <= q_low < q_high <= 1.):
            raise ValueError("q must be a pair (q_low, q_high) with "
                             "0 <= q_low < q_high <= 1")
        Y = np.asarray(self.luminance_of_spectrum(data)).ravel()
        positive = Y > 0.
        n_dropped = Y.size - int(np.count_nonzero(positive))
        if n_dropped == Y.size:
            raise ValueError("no positive luminances in the given data")
        if n_dropped > 0.1*Y.size:
            logger.warning(
                "luminance_quantiles: %d of %d pixels (%.0f%%) have non-positive "
                "luminance and were excluded from the quantile computation",
                n_dropped, Y.size, 100.*n_dropped/Y.size)
        return tuple(float(v) for v in np.quantile(Y[positive], (q_low, q_high)))

    # --- Display transform specification ---

    def set_luminance_range(self, *, Y_saturation, Y_black=None, dynamic_range=None,
                            highlights='clamp'):
        """Fix the luminance range that is mapped onto the displayable range.

        Luminance ``Y_black`` is rendered black, luminance ``Y_saturation`` is
        rendered at full display brightness, and the tone curve in between is linear
        unless :meth:`use_log_compression` has been enabled.

        Note that ``Y_saturation`` is deliberately *not* called a white point: a
        pixel reaching it is rendered at maximum brightness **for its own
        chromaticity**, which is white only if that chromaticity happens to be the
        D65 white point.  ``Y_black`` is unconditional by contrast, since zero
        luminance is black for every chromaticity.

        The arguments are keyword-only on purpose: :meth:`luminance_quantiles`
        returns ``(low, high)`` while this method reads ``Y_saturation`` first, so a
        positional API would make ``set_luminance_range(*proj.luminance_quantiles(x))``
        silently swap the two ends of the range.

        Parameters
        ----------
        Y_saturation : float
            Luminance rendered at full display brightness. Typically obtained from
            :meth:`luminance_of_spectrum` or :meth:`luminance_quantiles`.
        Y_black : float or None
            Luminance rendered black. ``None`` (default) means no black point,
            i.e. 0. Mutually exclusive with ``dynamic_range``.
        dynamic_range : float or None
            Alternative way to place the black point, as
            ``Y_saturation/dynamic_range``. Mutually exclusive with ``Y_black``.
        highlights : {'clamp', 'clip_channels'}
            What happens above ``Y_saturation``.

            - ``'clamp'`` *(default)*: luminance is clamped, which preserves
              chromaticity — over-bright pixels keep their hue and lose detail.
            - ``'clip_channels'``: luminance is left unclamped and the individual
              sRGB channels clip independently, which shifts hue as a pixel blows
              out, the way a camera sensor does.
        """
        Y_saturation = float(self._check_pos_scalar(Y_saturation, "Y_saturation"))
        if Y_black is not None and dynamic_range is not None:
            raise ValueError("give at most one of Y_black and dynamic_range — they are "
                             "two ways of specifying the same quantity")
        if dynamic_range is not None:
            dynamic_range = float(self._check_pos_scalar(dynamic_range, "dynamic_range"))
            if dynamic_range <= 1.:
                raise ValueError("dynamic_range must be > 1")
            Y_black = Y_saturation/dynamic_range
        elif Y_black is None:
            Y_black = 0.
        else:
            if not self._is_scalar(Y_black):
                raise ValueError("Y_black must be a scalar")
            Y_black = float(Y_black)
            if Y_black < 0.:
                raise ValueError("Y_black must be non-negative")
        if Y_black >= Y_saturation:
            raise ValueError("Y_black must be smaller than Y_saturation (got "
                             f"Y_black={Y_black}, Y_saturation={Y_saturation})")
        if highlights not in self._HIGHLIGHT_MODES:
            raise ValueError(f"highlights must be one of {self._HIGHLIGHT_MODES}")

        self._Y_black = Y_black
        self._Y_saturation = Y_saturation
        self._highlights = highlights

    def use_log_compression(self, enabled=True):
        """Switch logarithmic compression of the luminance range on or off.

        With log compression the tone curve is
        ``log(Y/Y_black)/log(Y_saturation/Y_black)`` instead of the linear
        ``(Y - Y_black)/(Y_saturation - Y_black)``.  A black point is then mandatory;
        projecting without one raises.
        """
        self._log_compression = bool(enabled)

    @property
    def luminance_range(self):
        """(Y_black, Y_saturation) pair, or None if it was never set."""
        if self._Y_saturation is None:
            return None
        return (self._Y_black, self._Y_saturation)

    @property
    def flux_convention(self):
        """Physical convention of the data this projector expects."""
        return self._flux_convention

    # --- Projection ---

    def project(self, data, XYZ_inspect_callback=None):
        """Project spatio-spectral data to sRGB perceived colors.

        The data is interpreted according to this projector's ``flux_convention``.

        Parameters
        ----------
        data : array-like
            Data with the spectral axis last.
        XYZ_inspect_callback : callable or None
            Called as ``fn(XYZ_raw, XYZ_tone_mapped)`` for inspection. Default: None.

        Returns
        -------
        numpy.ndarray
            sRGB values in [0, 1]; last axis has length 3.
        """
        data = np.asarray(data, dtype=float)
        self._pre_projection_checks(data)

        XYZ_data = self._transform_input_flux_to_XYZ(data)
        Y = XYZ_data[..., 1]

        if self._Y_saturation is None:
            Y_saturation = float(np.max(Y))
            Y_black = 0.
            saturation_origin = "auto, from data maximum"
            if not self._auto_saturation_warned:
                self._auto_saturation_warned = True
                logger.warning(
                    "No luminance range set: Y_saturation is taken from the maximum "
                    "luminance of each projected image, so renderings from separate "
                    "calls are NOT comparable with each other. Call "
                    "set_luminance_range(Y_saturation=..., Y_black=...) to fix the "
                    "mapping.")
        else:
            Y_black, Y_saturation = self._Y_black, self._Y_saturation
            saturation_origin = "explicit"

        if self._log_compression and Y_black <= 0.:
            raise ValueError(
                "log compression requires a black point, but none is set. Pass "
                "Y_black=... or dynamic_range=... to set_luminance_range().")
        if Y_saturation <= Y_black:
            if Y_saturation == 0.:
                # all-zero input under an auto saturation luminance: nothing to do
                return np.zeros(data.shape[:-1] + (3,))
            raise ValueError("Y_saturation must exceed Y_black")

        logger.info("project(): Y_black=%.4g Y_saturation=%.4g (%s), curve=%s, "
                    "highlights=%s", Y_black, Y_saturation, saturation_origin,
                    "log" if self._log_compression else "linear", self._highlights)

        self._last_transform = (Y_black, Y_saturation, self._log_compression,
                                self._highlights)
        self._last_flux_range = self._flux_range_of(data)

        XYZ_mapped = self._apply_tone_curve(XYZ_data, Y_black, Y_saturation,
                                            self._log_compression, self._highlights)

        if XYZ_inspect_callback is not None:
            if not callable(XYZ_inspect_callback):
                raise TypeError("XYZ_inspect_callback must be callable")
            XYZ_inspect_callback(XYZ_data, XYZ_mapped)

        return ColorSpaceTools.embed_XYZ_perceived_color_in_sRGB(XYZ_mapped)

    # --- Tone mapping ---

    def _apply_tone_curve(self, XYZ_data, Y_black, Y_saturation, log_compression,
                          highlights):
        """Map luminance onto [0, 1] while preserving chromaticity.

        All three XYZ channels are scaled by ``L/Y``, so only brightness changes.
        The black end always clamps at 0; the white end clamps at 1 only for
        ``highlights='clamp'``.
        """
        Y = XYZ_data[..., 1]
        if log_compression:
            L = np.log(np.maximum(Y, Y_black)/Y_black)/np.log(Y_saturation/Y_black)
        else:
            L = (Y - Y_black)/(Y_saturation - Y_black)
        L = np.maximum(L, 0.)
        if highlights == 'clamp':
            L = np.minimum(L, 1.)
        scale = np.where(Y > 0., L/np.where(Y > 0., Y, 1.), 0.)
        return XYZ_data*scale[..., np.newaxis]

    def _flux_range_of(self, data):
        """Record (min, smallest positive, max) of the fluxes actually seen."""
        positive = data[data > 0.]
        min_positive = float(np.min(positive)) if positive.size else None
        return (float(np.min(data)), min_positive, float(np.max(data)))

    # --- Colour map ---

    def get_color_map_image(self, levels=None, n_levels=128):
        """Return an sRGB image showing the color assigned to each spectral bin.

        Produces a 2-D image of shape ``(n_levels, n_bins, 3)``. Row ``i``, column
        ``k`` shows how a monochromatic spectrum carrying flux ``levels[i]`` in bin
        ``k`` alone is rendered — nothing more and nothing less.

        Note that no per-column rescaling takes place: the luminance produced per
        unit flux varies by roughly a factor of ten across the visible range, so a
        mid-spectrum column genuinely is much brighter than a column at either end
        at the same flux, and columns reach full brightness at different heights.
        That is what the projection does, so that is what this legend shows.

        The tone curve is taken from the most recent :meth:`project` call (falling
        back to an explicitly set luminance range), which is what keeps this legend
        consistent with the image it accompanies.

        Parameters
        ----------
        levels : array-like, 1-D, or None
            Flux levels to sample, in this projector's ``flux_convention``.
            ``None`` (default) spans the flux range of the most recently projected
            data with ``n_levels`` samples, and warns — the resulting levels are
            rarely the round, domain-appropriate numbers a reader wants.
        n_levels : int
            Number of rows generated when ``levels`` is None. Default: 128.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_levels, n_bins, 3)`` of sRGB values in [0, 1].
        """
        Y_black, Y_saturation, log_compression, highlights = self._effective_transform()
        self._ensure_mapping_tensor()

        if levels is None:
            levels = self._default_color_map_levels(n_levels, log_compression)
        levels = np.asarray(levels, dtype=float)
        if levels.ndim != 1:
            raise ValueError("levels must be a 1-D array")
        if levels.size == 0:
            raise ValueError("levels must not be empty")
        if np.any(levels < 0.):
            raise ValueError("levels must be non-negative")

        n_bins = len(self._input_spectrum_bin_lower)
        probe = np.eye(n_bins)[np.newaxis, :, :]*levels[:, np.newaxis, np.newaxis]
        XYZ = self._transform_input_flux_to_XYZ(probe)
        XYZ = self._apply_tone_curve(XYZ, Y_black, Y_saturation, log_compression,
                                     highlights)
        self._last_color_map_levels = levels
        return ColorSpaceTools.embed_XYZ_perceived_color_in_sRGB(XYZ)

    def _effective_transform(self):
        """The tone curve the colour map should use: last projected, else explicit."""
        if self._last_transform is not None:
            return self._last_transform
        if self._Y_saturation is None:
            raise RuntimeError(
                "no tone curve available — call project() or set_luminance_range() "
                "before generating a colour map")
        if self._log_compression and self._Y_black <= 0.:
            raise ValueError(
                "log compression requires a black point, but none is set. Pass "
                "Y_black=... or dynamic_range=... to set_luminance_range().")
        return (self._Y_black, self._Y_saturation,
                self._log_compression, self._highlights)

    def _default_color_map_levels(self, n_levels, log_compression):
        if self._last_flux_range is None:
            raise RuntimeError(
                "colour map levels can only be derived from previously projected data "
                "— call project() first, or pass levels= explicitly")
        n_levels = int(n_levels)
        if n_levels < 2:
            raise ValueError("n_levels must be at least 2")
        flux_min, flux_min_positive, flux_max = self._last_flux_range
        logger.warning(
            "get_color_map_image(): levels default to the flux range of the last "
            "projected data; pass levels= to label the colour map with round, "
            "domain-appropriate flux values.")
        if log_compression:
            if flux_min_positive is None:
                raise ValueError("no positive fluxes in the last projected data")
            return np.geomspace(flux_min_positive, flux_max, n_levels)
        return np.linspace(flux_min, flux_max, n_levels)

    def get_color_map_ticks(self, n_ticks=5, extent=None):
        """Return tick positions and labels for the last generated colour map.

        Picks round flux values inside the range of the levels most recently passed
        to (or generated by) :meth:`get_color_map_image` and returns where they sit
        along the level axis.  ``n_ticks`` is a *target*, not a guarantee: round
        numbers do not generally land exactly ``n_ticks`` times in a given range.

        Parameters
        ----------
        n_ticks : int
            Desired number of ticks. Default: 5.
        extent : pair of float or None
            Data coordinates of the first and last level of the colour map image.
            ``None`` (default) assumes ``[0, n_levels-1]``, i.e. pixel indices.

        Returns
        -------
        (numpy.ndarray, list of str)
            Tick positions in data coordinates, and their formatted labels.
        """
        levels = self._last_color_map_levels
        if levels is None:
            raise RuntimeError("call get_color_map_image() before requesting ticks")
        n_ticks = int(n_ticks)
        if n_ticks < 2:
            raise ValueError("n_ticks must be at least 2")

        lo, hi = float(levels[0]), float(levels[-1])
        log_spaced = lo > 0. and self._last_color_map_is_log_spaced(levels)
        values = self._round_tick_values(lo, hi, n_ticks, log_spaced)
        if len(values) == 0:
            values = np.array([lo, hi])

        if extent is None:
            extent = (0., float(len(levels) - 1))
        start, stop = float(extent[0]), float(extent[1])
        idx = np.interp(values, levels, np.arange(len(levels), dtype=float))
        if len(levels) > 1:
            positions = start + idx*(stop - start)/(len(levels) - 1)
        else:
            positions = np.full(len(values), start)
        return positions, [self._format_tick(v) for v in values]

    def apply_color_map_ticks(self, ax, axis='y', n_ticks=5):
        """Set round-numbered flux ticks on a matplotlib axis showing a colour map.

        Reads the image extent back off ``ax``, so whatever ``extent`` and
        ``origin`` were used when displaying the colour map are respected.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis that the colour map image was drawn into.
        axis : {'y', 'x'}
            Which axis the level (flux) dimension was drawn along. Default: 'y'.
        n_ticks : int
            Desired number of ticks. Default: 5.
        """
        if axis not in ('x', 'y'):
            raise ValueError("axis must be 'x' or 'y'")
        images = ax.get_images()
        if not images:
            raise RuntimeError("the given axis contains no image")
        left, right, bottom, top = images[-1].get_extent()
        extent = (bottom, top) if axis == 'y' else (left, right)
        positions, labels = self.get_color_map_ticks(n_ticks=n_ticks, extent=extent)
        if axis == 'y':
            ax.set_yticks(positions)
            ax.set_yticklabels(labels)
        else:
            ax.set_xticks(positions)
            ax.set_xticklabels(labels)
        return positions, labels

    @staticmethod
    def _last_color_map_is_log_spaced(levels):
        if len(levels) < 3 or levels[0] <= 0.:
            return False
        ratios = levels[1:]/levels[:-1]
        return bool(np.allclose(ratios, ratios[0]))

    @staticmethod
    def _round_tick_values(lo, hi, n_target, log_spaced):
        """Round numbers inside [lo, hi], aiming for roughly n_target of them."""
        if hi <= lo:
            return np.array([lo])
        if log_spaced:
            exp_lo, exp_hi = np.log10(lo), np.log10(hi)
            for mantissas in ([1.], [1., 3.], [1., 2., 5.], [1., 2., 3., 5., 7.]):
                candidates = np.array(
                    [m*10.**e for e in np.arange(np.floor(exp_lo), np.ceil(exp_hi) + 1)
                     for m in mantissas])
                candidates = np.sort(candidates[(candidates >= lo) & (candidates <= hi)])
                if len(candidates) >= n_target:
                    return candidates
            return candidates
        raw_step = (hi - lo)/(n_target - 1)
        magnitude = 10.**np.floor(np.log10(raw_step))
        for factor in (1., 2., 2.5, 5., 10.):
            step = factor*magnitude
            if step >= raw_step:
                break
        first = np.ceil(lo/step)*step
        return np.arange(first, hi + 0.5*step, step)

    @staticmethod
    def _format_tick(value):
        if value == 0.:
            return "0"
        exponent = np.log10(abs(value))
        if -3. <= exponent < 4.:
            return f"{value:g}"
        return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e")

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

        In both cases the normalised position ``t`` is affine in the input
        coordinate (``'proportional'``) or in the bin index (``'uniform'``), and the
        wavelength is affine in ``t``.  The result is therefore a rescaling onto the
        configured window, not a physical energy-to-wavelength conversion.

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

    def _ensure_mapping_tensor(self):
        if self._input_spectrum_bin_widths is None:
            raise RuntimeError("specify spectral bins first")
        if self._visible_spectrum_bin_flux_to_XYZ_mapping_tensor is None:
            self._generate_visible_spectrum_bin_flux_to_XYZ_mapping_tensor()

    def _transform_input_flux_to_XYZ(self, data):
        """Convert input data (in this projector's flux convention) to XYZ."""
        if self._flux_convention == 'flux_density':
            broadcast_sl = (None,)*(data.ndim - 1) + (slice(None),)
            data = data*self._input_spectrum_bin_widths[broadcast_sl]
        return np.tensordot(data,
                            self._visible_spectrum_bin_flux_to_XYZ_mapping_tensor,
                            axes=(-1, 0))

    # --- Pre-flight checks ---

    def _check_spectral_shape(self, data):
        if self._input_spectrum_bin_widths is None:
            raise RuntimeError("specify spectral bins before projecting")
        if data.ndim == 0 or data.shape[-1] != self._input_spectrum_bin_lower.shape[0]:
            raise ValueError("last dimension of input does not match number of "
                             "spectral bins")

    def _pre_projection_checks(self, input_data):
        self._check_spectral_shape(input_data)
        if np.any(input_data < 0.):
            raise ValueError("fluxes and flux densities must be non-negative")
        self._ensure_mapping_tensor()

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
