# -*- coding: utf-8 -*-

import numpy as np


def bare_dot(a, b):
    try:
        return a.dot(b, bare=True)
    except(AttributeError, TypeError):
        pass

    try:
        return a.vdot(b)
    except(AttributeError):
        pass

    return np.vdot(a, b)
