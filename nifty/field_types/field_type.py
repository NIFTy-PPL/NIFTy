# -*- coding: utf-8 -*-

from nifty.domain_object import DomainObject


class FieldType(DomainObject):

    def weight(self, x, power=1, axes=None, inplace=False):
        if inplace:
            result = x
        else:
            result = x.copy()
        return result

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
