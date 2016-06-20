# -*- coding: utf-8 -*-


class Base_field_type(object):
    def __init__(self, shape):
        self.shape = shape

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        try:
            new_shape = tuple([int(i) for i in shape])
        except TypeError:
            new_shape = (int(shape), )
        self._shape = new_shape

    def process(self, method_name, array, inplace=True, **kwargs):
        try:
            result_array = self.__getattr__(method_name)(array,
                                                         inplace,
                                                         **kwargs)
        except AttributeError:
            if inplace:
                result_array = array
            else:
                result_array = array.copy()

        return result_array
