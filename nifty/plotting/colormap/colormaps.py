from nifty.plotting.colormap.colormap import Colormap


def HighEnergyCmap():
    """
        Returns a color map often used in High Energy Astronomy.
    """

    red = [(0.0, 0.0),
           (0.167, 0.0),
           (0.333, 0.5),
           (0.5, 1.0),
           (0.667, 1.0),
           (0.833, 1.0),
           (1.0, 1.0)]

    green = [(0.0, 0.0),
             (0.167, 0.0),
             (0.333, 0.0),
             (0.5, 0.0),
             (0.667, 0.5),
             (0.833, 1.0),
             (1.0, 1.0)]

    blue = [(0.0, 0.0),
            (0.167, 1.0),
            (0.333, 0.5),
            (0.5, 0.0),
            (0.667, 0.0),
            (0.833, 0.0),
            (1.0, 1.0)]
    return Colormap("High Energy", red, green, blue)


def FaradayMapCmap():
    """
        Returns a color map used in reconstruction of the "Faraday Map".

        References
        ----------
        .. [#] N. Opermann et. al.,
            "An improved map of the Galactic Faraday sky",
            Astronomy & Astrophysics, Volume 542, id.A93, 06/2012;
            `arXiv:1111.6186 <http://www.arxiv.org/abs/1111.6186>`_
    """
    red = [(0.0, 0.35),
           (0.1, 0.4),
           (0.2, 0.25),
           (0.41, 0.47),
           (0.5, 0.8),
           (0.56, 0.96),
           (0.59, 1.0),
           (0.74, 0.8),
           (0.8, 0.8),
           (0.9, 0.5),
           (1.0, 0.4)]

    green = [(0.0, 0.0),
             (0.2, 0.0),
             (0.362, 0.88),
             (0.5, 1.0),
             (0.638, 0.88),
             (0.8, 0.25),
             (0.9, 0.3),
             (1.0, 0.2)]

    blue = [(0.0, 0.35),
            (0.1, 0.4),
            (0.2, 0.8),
            (0.26, 0.8),
            (0.41, 1.0),
            (0.44, 0.96),
            (0.5, 0.8),
            (0.59, 0.47),
            (0.8, 0.0),
            (1.0, 0.0)]

    return Colormap("Faraday Map", red, green, blue)


def FaradayUncertaintyCmap():
    """
        Returns a color map used for the "Faraday Map Uncertainty".

        References
        ----------
        .. [#] N. Opermann et. al.,
            "An improved map of the Galactic Faraday sky",
            Astronomy & Astrophysics, Volume 542, id.A93, 06/2012;
            `arXiv:1111.6186 <http://www.arxiv.org/abs/1111.6186>`_
    """
    red = [(0.0, 1.0),
           (0.1, 0.8),
           (0.2, 0.65),
           (0.41, 0.6),
           (0.5, 0.7),
           (0.56, 0.96),
           (0.59, 1.0),
           (0.74, 0.8),
           (0.8, 0.8),
           (0.9, 0.5),
           (1.0, 0.4)]

    green = [(0.0, 0.9),
             (0.2, 0.65),
             (0.362, 0.95),
             (0.5, 1.0),
             (0.638, 0.88),
             (0.8, 0.25),
             (0.9, 0.3),
             (1.0, 0.2)]

    blue = [(0.0, 1.0),
            (0.1, 0.8),
            (0.2, 1.0),
            (0.41, 1.0),
            (0.44, 0.96),
            (0.5, 0.7),
            (0.59, 0.42),
            (0.8, 0.0),
            (1.0, 0.0)]

    return Colormap("Faraday Uncertainty", red, green, blue)


def PlusMinusCmap():
    """
        Returns a color map useful for a zero-centerd range of values.
    """
    red = [(0.0, 1.0),
           (0.1, 0.96),
           (0.2, 0.84),
           (0.3, 0.64),
           (0.4, 0.36),
           (0.5, 0.0),
           (0.6, 0.0),
           (0.7, 0.0),
           (0.8, 0.0),
           (0.9, 0.0),
           (1.0, 0.0)]

    green = [(0.0, 0.5),
             (0.1, 0.32),
             (0.2, 0.18),
             (0.3, 0.08),
             (0.4, 0.02),
             (0.5, 0.0),
             (0.6, 0.02),
             (0.7, 0.08),
             (0.8, 0.18),
             (0.9, 0.32),
             (1.0, 0.5)]

    blue = [(0.0, 0.0),
            (0.1, 0.0),
            (0.2, 0.0),
            (0.3, 0.0),
            (0.4, 0.0),
            (0.5, 0.0),
            (0.6, 0.36),
            (0.7, 0.64),
            (0.8, 0.84),
            (0.9, 0.96),
            (1.0, 1.0)]

    return Colormap("Plus Minus", red, green, blue)


def PlankCmap():
    """
        Returns a color map similar to the one used for the "Planck CMB Map".
    """
    red = [(0.0, 0.0),
           (0.1, 0.0),
           (0.2, 0.0),
           (0.3, 0.0),
           (0.4, 0.0),
           (0.5, 1.0),
           (0.6, 1.0),
           (0.7, 1.0),
           (0.8, 0.83),
           (0.9, 0.67),
           (1.0, 0.5)]

    green = [(0.0, 0.0),
             (0.1, 0.0),
             (0.2, 0.0),
             (0.3, 0.3),
             (0.4, 0.7),
             (0.5, 1.0),
             (0.6, 0.7),
             (0.7, 0.3),
             (0.8, 0.0),
             (0.9, 0.0),
             (1.0, 0.0)]

    blue = [(0.0, 0.5),
            (0.1, 0.67),
            (0.2, 0.83),
            (0.3, 1.0),
            (0.4, 1.0),
            (0.5, 1.0),
            (0.6, 0.0),
            (0.7, 0.0),
            (0.8, 0.0),
            (0.9, 0.0),
            (1.0, 0.0)]

    return Colormap("Planck-like", red, green, blue)
