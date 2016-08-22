# -*- coding: utf-8 -*-

import numpy as np

class Random(object):
    @staticmethod
    def pm1(dtype=np.dtype('int'), shape=1):

        size = int(np.prod(shape))
        if issubclass(dtype.type, np.complexfloating):
            x = np.array([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=dtype)
            x = x[np.random.randint(4, high=None, size=size)]
        else:
            x = 2 * np.random.randint(2, high=None, size=size) - 1

        return x.astype(dtype).reshape(shape)

    @staticmethod
    def normal(dtype=np.dtype('float64'), shape=(1,), mean=None, std=None):

        size = int(np.prod(shape))
        if issubclass(dtype.type, np.complexfloating):
            x = np.empty(size, dtype=dtype)
            x.real = np.random.normal(loc=0, scale=np.sqrt(0.5), size=size)
            x.imag = np.random.normal(loc=0, scale=np.sqrt(0.5), size=size)
        else:
            x = np.random.normal(loc=0, scale=1, size=size)
            x = x.astype(dtype, copy=False)

        x = x.reshape(shape)

        if std is not None:
            x *= dtype.type(std)

        if mean is not None:
            x += dtype.type(mean)

        return x

    @staticmethod
    def uniform(dtype=np.dtype('float64'), shape=1, low=0, high=1):

        size = int(np.prod(shape))
        if issubclass(dtype.type, np.complexfloating):
            x = np.empty(size, dtype=dtype)
            x.real = (high - low) * np.random.random(size=size) + low
            x.imag = (high - low) * np.random.random(size=size) + low
        elif dtype in [np.dtype('int8'), np.dtype('int16'), np.dtype('int32'),
                       np.dtype('int64')]:
            x = np.random.random_integers(min(low, high),
                                          high=max(low, high),
                                          size=size)
        else:
            x = (high - low) * np.random.random(size=size) + low

        return x.astype(dtype, copy=False).reshape(shape)
