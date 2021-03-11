from jax.numpy import isscalar, array
from jax.tree_util import tree_flatten

def makeField(obj):
    if isscalar(obj):
        return obj
    val, domain = tree_flatten(obj)
    from .field import Field
    return Field(domain, [array(v) for v in val])

def fromField(f):
    if isscalar(f):
        return f
    return f.to_tree()

def just_add(a,b):
    return fromField(makeField(a) + makeField(b))


