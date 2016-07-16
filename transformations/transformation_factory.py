import numpy as np

from nifty.rg import RGSpace
from nifty.lm import GLSpace, HPSpace, LMSpace

import transformation


class TransformationFactory(object):
    """
        Transform factory which generates transform objects
    """

    def __init__(self):
        # cache for storing the transform objects
        self.cache = {}

    def _get_transform(self, domain, codomain, module):
        if isinstance(domain, RGSpace):
            return transformation.RGRGTransformation(domain, codomain, module)
        elif isinstance(domain, GLSpace):
            return transformation.GLLMTransformation(domain, codomain, module)
        elif isinstance(domain, HPSpace):
            return transformation.HPLMTransformation(domain, codomain, module)
        elif isinstance(domain, LMSpace):
            if isinstance(codomain, GLSpace):
                return transformation.LMGLTransformation(domain, codomain,
                                                         module)
            elif isinstance(codomain, HPSpace):
                return transformation.LMHPTransformation(domain, codomain,
                                                         module)
            else:
                raise ValueError('ERROR: incompatible codomain')
        else:
            raise ValueError('ERROR: unknown domain')

    def create(self, domain, codomain, module=None):
        key = domain.__hash__() ^ ((111 * codomain.__hash__()) ^
                                   (179 * module.__hash__()))

        if key not in self.cache:
            self.cache[key] = self._get_transform(domain, codomain, module)

        return self.cache[key]