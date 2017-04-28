# -*- coding: utf-8 -*-

import os

__all__ = []

try:
    import matplotlib
except ImportError:
    pass
else:
    try:
        display = os.environ['DISPLAY']
    except KeyError:
        matplotlib.use('Agg')
    else:
        if display == '':
            matplotlib.use('Agg')
