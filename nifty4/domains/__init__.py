"""
Sample documentation for spaces. (or any other package) This will be displayed under Summary section of module documentation
"""

from .domain import Domain
from .unstructured_domain import UnstructuredDomain
from .structured_domain import StructuredDomain
from .rg_space import RGSpace
from .lm_space import LMSpace
from .hp_space import HPSpace
from .gl_space import GLSpace
from .dof_space import DOFSpace
from .power_space import PowerSpace

__all__ = ["Domain", "UnstructuredDomain", "StructuredDomain", "RGSpace",
           "LMSpace", "HPSpace", "GLSpace", "DOFSpace", "PowerSpace"]
