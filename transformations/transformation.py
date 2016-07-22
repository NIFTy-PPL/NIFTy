class Transformation(object):
    """
        A generic transformation which defines a static check_codomain
        method for all transforms.
    """

    def __init__(self, domain, codomain=None, module=None):
        pass

    def transform(self, val, axes=None, **kwargs):
        raise NotImplementedError
