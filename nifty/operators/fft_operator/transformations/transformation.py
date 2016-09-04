
import abc


class Transformation(object):
    """
        A generic transformation which defines a static check_codomain
        method for all transforms.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, domain, codomain=None, module=None):
        if codomain is None:
            self.domain = domain
            self.codomain = self.get_codomain(domain)
        elif self.check_codomain(domain, codomain):
            self.domain = domain
            self.codomain = codomain
        else:
            raise ValueError("ERROR: Incompatible codomain!")

    @staticmethod
    def get_codomain(domain, dtype=None, zerocenter=None, **kwargs):
        raise NotImplementedError

    @staticmethod
    def check_codomain(domain, codomain):
        raise NotImplementedError

    def transform(self, val, axes=None, **kwargs):
        raise NotImplementedError
