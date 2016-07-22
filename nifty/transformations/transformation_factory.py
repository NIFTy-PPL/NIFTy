from nifty.spaces import RGSpace, GLSpace, HPSpace, LMSpace

from rgrgtransformation import RGRGTransformation
from gllmtransformation import GLLMTransformation
from hplmtransformation import HPLMTransformation
from lmgltransformation import LMGLTransformation
from lmhptransformation import LMHPTransformation


class _TransformationFactory(object):
    """
        Transform factory which generates transform objects
    """

    def __init__(self):
        # cache for storing the transform objects
        self.cache = {}

    def _get_transform(self, domain, codomain, module):
        if isinstance(domain, RGSpace):
            if isinstance(codomain, RGSpace):
                return RGRGTransformation(domain, codomain, module)
            else:
                raise ValueError('ERROR: incompatible codomain')

        elif isinstance(domain, GLSpace):
            if isinstance(codomain, GLSpace):
                return GLLMTransformation(domain, codomain, module)
            else:
                raise ValueError('ERROR: incompatible codomain')

        elif isinstance(domain, HPSpace):
            if isinstance(codomain, GLSpace):
                return HPLMTransformation(domain, codomain, module)
            else:
                raise ValueError('ERROR: incompatible codomain')

        elif isinstance(domain, LMSpace):
            if isinstance(codomain, GLSpace):
                return LMGLTransformation(domain, codomain, module)
            elif isinstance(codomain, HPSpace):
                return LMHPTransformation(domain, codomain, module)
            else:
                raise ValueError('ERROR: incompatible codomain')
        else:
            raise ValueError('ERROR: unknown domain')

    def create(self, domain, codomain, module=None):
        key = domain.__hash__() ^ ((codomain.__hash__()/111) ^
                                   (module.__hash__())/179)

        if key not in self.cache:
            self.cache[key] = self._get_transform(domain, codomain, module)

        return self.cache[key]


TransformationFactory = _TransformationFactory()
