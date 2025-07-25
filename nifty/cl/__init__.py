from .. import config
from . import random

from .logger import logger, logger_init
from .utilities import memo, frozendict, myassert, device_available

from .domains import *
from .domain_tuple import DomainTuple
from .multi_domain import MultiDomain

from .operators.operator import (
    Operator, is_likelihood_energy, is_operator, is_linearization, is_fieldlike
)
from .operators.linear_operator import LinearOperator

from .any_array import AnyArray
from .field import Field
from .multi_field import MultiField


from .operators.adder import Adder
from .operators.diagonal_operator import DiagonalOperator
from .operators.distributors import DOFDistributor, PowerDistributor
from .operators.domain_tuple_field_inserter import DomainTupleFieldInserter
from .operators.einsum import LinearEinsum, MultiLinearEinsum
from .operators.contraction_operator import ContractionOperator, IntegrationOperator
from .operators.linear_interpolation import LinearInterpolator
from .operators.endomorphic_operator import EndomorphicOperator
from .operators.harmonic_operators import (
    FFTOperator, HartleyOperator, SHTOperator, HarmonicTransformOperator,
    HarmonicSmoothingOperator)
from .operators.field_zero_padder import FieldZeroPadder
from .operators.inversion_enabler import InversionEnabler
from .operators.mask_operator import MaskOperator
from .operators.regridding_operator import RegriddingOperator
from .operators.sampling_enabler import SamplingEnabler
from .operators.sandwich_operator import SandwichOperator
from .operators.scaling_operator import ScalingOperator
from .operators.selection_operators import SliceOperator, SplitOperator
from .operators.block_diagonal_operator import BlockDiagonalOperator
from .operators.outer_product_operator import OuterProduct
from .operators.simple_linear_operators import (
    VdotOperator, ConjugationOperator, Realizer, FieldAdapter, ducktape,
    GeometryRemover, NullOperator, PartialExtractor, Imaginizer, PrependKey,
    DomainChangerAndReshaper, ExtractAtIndices, SqueezeOperator, Variable)
from .operators.matrix_product_operator import MatrixProductOperator
from .operators.value_inserter import ValueInserter
from .operators.energy_operators import (
    EnergyOperator, GaussianEnergy, PoissonianEnergy, InverseGammaEnergy,
    BernoulliEnergy, StandardHamiltonian, AveragedEnergy, QuadraticFormOperator,
    Squared2NormOperator, StudentTEnergy, VariableCovarianceGaussianEnergy,
    LikelihoodEnergyOperator)
from .operators.convolution_operators import FuncConvolutionOperator
from .operators.normal_operators import NormalTransform, LognormalTransform
from .operators.multifield2vector import Multifield2Vector
from .operators.jax_operator import *
from .operators.counting_operator import CountingOperator
from .operators.transpose_operator import TransposeOperator

from .probing import probe_with_posterior_samples, probe_diagonal, \
    StatCalculator, approximation2endo

from .minimization.line_search import LineSearch
from .minimization.iteration_controllers import (
    IterationController, GradientNormController, DeltaEnergyController,
    GradInfNormController, AbsDeltaEnergyController, StochasticAbsDeltaEnergyController)
from .minimization.minimizer import Minimizer
from .minimization.conjugate_gradient import ConjugateGradient
from .minimization.nonlinear_cg import NonlinearCG
from .minimization.descent_minimizers import (
    DescentMinimizer, SteepestDescent, VL_BFGS, L_BFGS, RelaxedNewton,
    NewtonCG)
from .minimization.stochastic_minimizer import ADVIOptimizer
from .minimization.scipy_minimizer import L_BFGS_B
from .minimization.energy import Energy
from .minimization.quadratic_energy import QuadraticEnergy
from .minimization.sample_list import SampleList, SampleListBase, ResidualSampleList
from .minimization.energy_adapter import EnergyAdapter, StochasticEnergyAdapter
from .minimization.kl_energies import SampledKLEnergy, SampledKLEnergyClass
from .minimization.optimize_kl import optimize_kl
from .minimization.config.optimize_kl_config import OptimizeKLConfig

from .sugar import *

from .plot import Plot

from .library.special_distributions import InverseGammaOperator, \
    UniformOperator, LaplaceOperator, LogInverseGammaOperator, \
    GammaOperator, BetaOperator
from .library.los_response import LOSResponse
from .library.dynamic_operator import (dynamic_operator,
                                       dynamic_lightcone_operator)
from .library.light_cone_operator import LightConeOperator

from .library.wiener_filter_curvature import WienerFilterCurvature
from .library.adjust_variances import (make_adjust_variances_hamiltonian,
                                       do_adjust_variances)
from .library.nft import Gridder, FinuFFT, Nufft
from .library.correlated_fields import CorrelatedFieldMaker
from .library.correlated_fields_simple import SimpleCorrelatedField
from .library.variational_models import MeanFieldVI, FullCovarianceVI

from . import extra


from .linearization import Linearization

from .operator_spectrum import operator_spectrum
from .evidence_lower_bound import estimate_evidence_lower_bound

from .operator_tree_optimiser import optimise_operator

from .ducc_dispatch import set_nthreads, nthreads


# We deliberately don't set __all__ here, because we don't want people to do a
# "from nifty import *"; that would swamp the global namespace.
