from .version import __version__

from . import dobj

from .domains.domain import Domain
from .domains.structured_domain import StructuredDomain
from .domains.unstructured_domain import UnstructuredDomain
from .domains.rg_space import RGSpace
from .domains.lm_space import LMSpace
from .domains.gl_space import GLSpace
from .domains.hp_space import HPSpace
from .domains.power_space import PowerSpace
from .domains.dof_space import DOFSpace
from .domains.log_rg_space import LogRGSpace

from .domain_tuple import DomainTuple
from .field import Field

from .nonlinearities import Exponential, Linear, PositiveTanh, Tanh

from .models.constant import Constant
from .models.linear_model import LinearModel
from .models.local_nonlinearity import (LocalModel, PointwiseExponential,
                                        PointwisePositiveTanh, PointwiseTanh)
from .models.model import Model
from .models.multi_model import MultiModel
from .models.variable import Variable

from .operators.central_zero_padder import CentralZeroPadder
from .operators.diagonal_operator import DiagonalOperator
from .operators.dof_distributor import DOFDistributor
from .operators.domain_distributor import DomainDistributor
from .operators.endomorphic_operator import EndomorphicOperator
from .operators.exp_transform import ExpTransform
from .operators.fft_operator import FFTOperator
from .operators.field_zero_padder import FieldZeroPadder
from .operators.hartley_operator import HartleyOperator
from .operators.harmonic_smoothing_operator import HarmonicSmoothingOperator
from .operators.geometry_remover import GeometryRemover
from .operators.harmonic_transform_operator import HarmonicTransformOperator
from .operators.inversion_enabler import InversionEnabler
from .operators.laplace_operator import LaplaceOperator
from .operators.linear_operator import LinearOperator
from .operators.mask_operator import MaskOperator
from .operators.multi_adaptor import MultiAdaptor
from .operators.null_operator import NullOperator
from .operators.power_distributor import PowerDistributor
from .operators.qht_operator import QHTOperator
from .operators.sampling_enabler import SamplingEnabler
from .operators.sandwich_operator import SandwichOperator
from .operators.scaling_operator import ScalingOperator
from .operators.selection_operator import SelectionOperator
from .operators.slope_operator import SlopeOperator
from .operators.smoothness_operator import SmoothnessOperator
from .operators.symmetrizing_operator import SymmetrizingOperator

from .probing.utils import probe_with_posterior_samples, probe_diagonal, \
    StatCalculator

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
from .minimization.l_bfgs import L_BFGS
from .minimization.relaxed_newton import RelaxedNewton
from .minimization.scipy_minimizer import (ScipyMinimizer, NewtonCG, L_BFGS_B,
                                           ScipyCG)
from .minimization.energy import Energy
from .minimization.quadratic_energy import QuadraticEnergy
from .minimization.line_energy import LineEnergy
from .minimization.energy_sum import EnergySum

from .sugar import *
from .plotting.plot import plot, plot_finish

from .library.amplitude_model import make_amplitude_model
from .library.gaussian_energy import GaussianEnergy
from .library.los_response import LOSResponse
from .library.inverse_gamma_model import InverseGammaModel
from .library.poissonian_energy import PoissonianEnergy
from .library.wiener_filter_curvature import WienerFilterCurvature
from .library.correlated_fields import (make_correlated_field,
                                        make_mf_correlated_field)
from .library.bernoulli_energy import BernoulliEnergy

from . import extra

from .utilities import memo, frozendict

from .logger import logger

from .multi.multi_domain import MultiDomain
from .multi.multi_field import MultiField
from .multi.block_diagonal_operator import BlockDiagonalOperator

from .energies.kl import SampledKullbachLeiblerDivergence
from .energies.hamiltonian import Hamiltonian

# We deliberately don't set __all__ here, because we don't want people to do a
# "from nifty5 import *"; that would swamp the global namespace.
