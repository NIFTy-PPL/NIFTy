# -*- coding: utf-8 -*-


from base_field_type import FieldType


class FieldArray(FieldType):
    @property
    def dim(self):
        return reduce(lambda x, y: x*y, self.shape)
