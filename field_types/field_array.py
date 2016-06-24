# -*- coding: utf-8 -*-


from base_field_type import Field_type


class Field_array(Field_type):
    def dot_contraction(self, x, axes):
        return x.sum(axis=axes)
