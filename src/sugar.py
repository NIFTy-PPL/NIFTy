from jax import numpy as np
from jax.tree_util import tree_flatten

def makeField(obj):
    if np.isscalar(obj):
        return obj
    val, domain = tree_flatten(obj)
    from .field import Field
    return Field(domain, val)

def fromField(f):
    if np.isscalar(f):
        return f
    return f.to_tree()

def just_add(a,b):
    return fromField(makeField(a) + makeField(b))


