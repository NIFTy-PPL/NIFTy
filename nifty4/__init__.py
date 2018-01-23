from .version import __version__

from . import dobj

from .domain_object import DomainObject

from .spaces.field_array import FieldArray
from .spaces.space import Space
from .spaces.rg_space import RGSpace
from .spaces.lm_space import LMSpace
from .spaces.hp_space import HPSpace
from .spaces.gl_space import GLSpace
from .spaces.dof_space import DOFSpace
from .spaces.power_space import PowerSpace

from .domain_tuple import DomainTuple

from .operators.linear_operator import LinearOperator
from .operators.endomorphic_operator import EndomorphicOperator
from .operators.scaling_operator import ScalingOperator
from .operators.diagonal_operator import DiagonalOperator
from .operators.fft_operator import FFTOperator
from .operators.fft_smoothing_operator import FFTSmoothingOperator
from .operators.direct_smoothing_operator import DirectSmoothingOperator
from .operators.response_operator import ResponseOperator
from .operators.laplace_operator import LaplaceOperator
from .operators.power_projection_operator import PowerProjectionOperator
from .operators.inversion_enabler import InversionEnabler

from .field import Field, sqrt, exp, log

from .probing.prober import Prober
from .probing.diagonal_prober_mixin import DiagonalProberMixin
from .probing.trace_prober_mixin import TraceProberMixin

from .minimization.line_search import LineSearch
from .minimization.line_search_strong_wolfe import LineSearchStrongWolfe
from .minimization.iteration_controller import IterationController
from .minimization.gradient_norm_controller import GradientNormController
from .minimization.minimizer import Minimizer
from .minimization.conjugate_gradient import ConjugateGradient
from .minimization.nonlinear_cg import NonlinearCG
from .minimization.descent_minimizer import DescentMinimizer
from .minimization.steepest_descent import SteepestDescent
from .minimization.vl_bfgs import VL_BFGS
from .minimization.relaxed_newton import RelaxedNewton
from .minimization.energy import Energy
from .minimization.quadratic_energy import QuadraticEnergy
from .minimization.line_energy import LineEnergy

from .sugar import *
from .plotting.plot import plot
from . import library

__all__= ["DomainObject", "FieldArray", "Space", "RGSpace", "LMSpace",
          "HPSpace", "GLSpace", "DOFSpace", "PowerSpace", "DomainTuple",
          "LinearOperator", "EndomorphicOperator", "ScalingOperator",
          "DiagonalOperator", "FFTOperator", "FFTSmoothingOperator",
          "DirectSmoothingOperator", "ResponseOperator", "LaplaceOperator",
          "PowerProjectionOperator", "InversionEnabler",
          "Field", "sqrt", "exp", "log",
          "Prober", "DiagonalProberMixin", "TraceProberMixin"]
