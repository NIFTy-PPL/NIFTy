# -*- coding: utf-8 -*-


from base_field_type import FieldType


class FieldArray(FieldType):
    def dot_contraction(self, x, axes):
        return x.sum(axis=axes)
