# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from os import makedirs
from os.path import isdir, join
from warnings import warn

from ..domain_tuple import DomainTuple
from ..multi_field import MultiField
from ..operators.energy_operators import StandardHamiltonian
from ..operators.operator import Operator
from ..operators.scaling_operator import ScalingOperator
from ..plot import Plot, plottable2D
from ..sugar import from_random
from ..utilities import Nop, check_MPI_equality, get_MPI_params_from_comm
from .energy_adapter import EnergyAdapter
from .iteration_controllers import IterationController
from .kl_energies import SampledKLEnergy
from .minimizer import Minimizer
from .sample_list import SampleList, SampleListBase

try:
    import h5py
except ImportError:
    h5py = False

try:
    import astropy
except ImportError:
    astropy = False


def optimize_kl(likelihood_energy,
                global_iterations,
                n_samples,
                kl_minimizer,
                sampling_iteration_controller,
                nonlinear_sampling_minimizer,
                constants=[],
                point_estimates=[],
                plottable_operators={},
                output_directory="nifty_optimize_kl_output",
                initial_position=None,
                initial_index=0,
                ground_truth_position=None,
                comm=None,
                overwrite=False,
                callback=None,
                plot_latent=False,
                save_strategy="last",
                return_final_position=False):
    """Provide potentially useful interface for standard KL minimization.

    The parameters `likelihood_energy`, `kl_minimizer`,
    `sampling_iteration_controller`, `nonlinear_sampling_minimizer`, `constants`,
    `point_estimates` and `comm` also accept a function as input that takes the
    index of the global iteration as input and return the respective value that
    should be used for that iteration.

    High-level pseudo code of the algorithm, that is implemented by `optimize_kl`:

    .. code-block:: none

        for ii in range(initial_index, initial_index+global_iterations):
            samples = Draw `n_samples` approximate samples(position,
                                                           likelihood_energy
                                                           sampling_iteration_controller,
                                                           nonlinear_sampling_minimizer,
                                                           point_estimates)
            position, samples = Optimize approximate KL(likelihood_energy, samples) with `kl_minimizer`(constants)
            Save intermediate results(samples)

    Parameters
    ----------
    likelihood_energy : Operator or callable
        Likelihood energy shall be used for inference. It is assumed that the
        definition space of this energy contains parameters that are a-priori
        standard normal distributed.
    global_iterations : int
        Number of resampling loops.
    n_samples : int or callable
        Number of samples used to sample Kullback-Leibler divergence. 0
        corresponds to maximum-a-posteriori.
    kl_minimizer : Minimizer or callable
        Controls the minimizer for the KL optimization.
    sampling_iteration_controller : IterationController or None or callable
        Controls the conjugate gradient for inverting the posterior metric.  If
        `None`, approximate posterior samples cannot be drawn. It is only
        suited for maximum-a-posteriori solutions.
    nonlinear_sampling_minimizer : Minimizer or None or callable
        Controls the minimizer for the non-linear geometric sampling to
        approximate the KL. Can be either None (then the MGVI algorithm is used
        instead of geoVI) or a Minimizer.
    constants : list or callable
        List of parameter keys that are kept constant during optimization.
        Default is no constants.
    point_estimates : list or callable
        List of parameter keys for which no samples are drawn, but that are
        (possibly) optimized for, corresponding to point estimates of these.
        Default is to draw samples for the complete domain.
    plottable_operators : dict
        Dictionary of operators that are plotted during the minimization. The
        key contains a string that serves as identifier.
    output_directory : str or None
        Directory in which all output files are saved. If None, no output is
        stored.  Default: "nifty_optimize_kl_output".
    initial_position : Field, MultiField or None
        Position in the definition space of `likelihood_energy` from which the
        optimization is started. If `None`, it starts at a random, normal
        distributed position with standard deviation 0.1. Default: None.
    initial_index : int
        Initial index that is used to enumerate the output files. May be used
        if `optimize_kl` is called multiple times. Default: 0.
    ground_truth_position : Field, MultiField or None
        Position in latent space that represents the ground truth. Used only in
        plotting. May be useful for validating algorithms.
    comm : MPI communicator or None
        MPI communicator for distributing samples over MPI tasks. If `None`,
        the samples are not distributed. Default: None.
    overwrite : bool
        Determine if existing directories and files are allowed to be
        overwritten. Default: False.
    callback : callable or None
        Function that is called after every global iteration. It can be either
        a function with one argument (then the latest sample list is passed) or
        a function with two arguments (in which case the latest sample list and
        the global iteration index are passed). Default: None.
    plot_latent : bool
        Determine if latent space shall be plotted or not. Default: False.
    save_strategy : str
        If "last", only the samples of the last global iteration are stored. If
        "all", all intermediate samples are written to disk. `save_strategy` is
        only applicable if `output_directory` is not None. Default: "last".
    return_final_position : bool
        Determine if the final position of the minimization shall be return.
        May be useful to feed it as `initial_position` into another
        `optimize_kl` call. Default: False.

    Returns
    -------
    kl : Energy

    mean : Field or MultiField (optional)

    Note
    ----
    If `h5py` is available, the output of all plotted operators is saved as hdf5
    file as well.  If `astropy` is available, the mean and standard deviation of
    all plotted operators, that have a single 2d-RGSpace as target, are exported
    as FITS files.

    Note
    ----
    This function comes with some MPI support. Generally, with the help of MPI
    samples are distributed over tasks.
    """
    from ..utilities import myassert
    from .descent_minimizers import DescentMinimizer

    if not isinstance(plottable_operators, dict):
        raise TypeError
    if len(set(["latent", "pickle"]) & set(plottable_operators.keys())) != 0:
        raise ValueError("The keys `latent` and `pickle` in `plottable_operators` are reserved.")
    if plot_latent:
        plottable_operators = plottable_operators.copy()
        plottable_operators["latent"] = ScalingOperator(likelihood_energy.domain, 1.)
    if not isinstance(initial_index, int):
        raise TypeError
    if save_strategy not in ["all", "last"]:
        raise ValueError("Save strategy '{save_strategy}' not supported.")

    likelihood_energy = _make_callable(likelihood_energy)
    kl_minimizer = _make_callable(kl_minimizer)
    sampling_iteration_controller = _make_callable(sampling_iteration_controller)
    nonlinear_sampling_minimizer = _make_callable(nonlinear_sampling_minimizer)
    constants = _make_callable(constants)
    point_estimates = _make_callable(point_estimates)
    n_samples = _make_callable(n_samples)
    comm = _make_callable(comm)
    if callback is None:
        callback = lambda x: None

    # Sanity check of input
    for iglobal in range(initial_index, global_iterations + initial_index):
        for (obj, cls) in [(likelihood_energy, Operator), (kl_minimizer, DescentMinimizer),
                           (nonlinear_sampling_minimizer, (DescentMinimizer, type(None))),
                           (constants, (list, tuple)), (point_estimates, (list, tuple)),
                           (n_samples, int)]:
            if not isinstance(obj(iglobal), cls):
                raise TypeError(f"{obj(iglobal)} is not instance of {cls}")

        if sampling_iteration_controller(iglobal) is None:
            myassert(n_samples(iglobal) == 0)
        else:
            myassert(isinstance(sampling_iteration_controller(iglobal), IterationController))
        myassert(likelihood_energy(iglobal).target is DomainTuple.scalar_domain())
        if not comm(iglobal) is None:
            try:
                import mpi4py
                myassert(isinstance(comm(iglobal), mpi4py.MPI.Intracomm))
            except ImportError:
                pass
    myassert(_number_of_arguments(callback) in [1, 2])
    # /Sanity check of input

    if not likelihood_energy(0).target is DomainTuple.scalar_domain():
        raise TypeError
    mean = initial_position
    if mean is None:
        mean = 0.1 * from_random(likelihood_energy(0).domain)
    dom = mean.domain
    mf_dom = isinstance(mean, MultiField)
    if ground_truth_position is not None:
        if ground_truth_position.domain is not dom:
            raise ValueError("Ground truth needs to have the same domain as `likelihood_energy`.")

    if output_directory is not None:
        if not overwrite and isdir(output_directory):
            raise RuntimeError(f"{output_directory} already exists. Please delete or set "
                                "`overwrite` to `True`.")
        if _MPI_master(comm(0)):
            makedirs(output_directory, exist_ok=overwrite)
            for subfolder in ["pickle"] + list(plottable_operators.keys()):
                makedirs(join(output_directory, subfolder), exist_ok=overwrite)

    for iglobal in range(initial_index, global_iterations + initial_index):
        ham = StandardHamiltonian(likelihood_energy(iglobal), sampling_iteration_controller(iglobal))
        minimizer = kl_minimizer(iglobal)
        mean_iter = mean.extract(ham.domain)

        # Distributing the domain of the likelihood is not supported (yet)
        check_MPI_equality(likelihood_energy(iglobal).domain, comm(iglobal))
        check_MPI_equality(mean.domain, comm(iglobal))
        check_MPI_equality(mean, comm(iglobal))  # FIXME Temporary because potentially expensive

        if n_samples(iglobal) == 0:
            e = EnergyAdapter(mean_iter, ham, constants=constants(iglobal),
                              want_metric=_want_metric(minimizer))
            if comm(iglobal) is None:
                e, _ = minimizer(e)
                mean = MultiField.union([mean, e.position]) if mf_dom else e.position
                sl = SampleList([mean])
            else:
                warn("Have detected MPI communicator for optimizing Hamiltonian. Will use only "
                     "the rank0 task and communicate the result afterwards to all other tasks.")
                if _MPI_master(comm(iglobal)):
                    e, _ = minimizer(e)
                    mean = MultiField.union([mean, e.position]) if mf_dom else e.position
                    sl = SampleList([mean], comm=comm(iglobal), domain=dom)
                else:
                    mean = None
                    sl = SampleList([], comm=comm(iglobal), domain=dom)
                comm(iglobal).Barrier()
                mean = comm(iglobal).bcast(mean, root=0)
        else:
            e = SampledKLEnergy(
                mean_iter,
                ham,
                n_samples(iglobal),
                nonlinear_sampling_minimizer(iglobal),
                comm=comm(iglobal),
                constants=constants(iglobal),
                point_estimates=point_estimates(iglobal))
            e, _ = minimizer(e)
            mean = MultiField.union([mean, e.position]) if mf_dom else e.position
            sl = e.samples

        if output_directory is not None:
            _plot_operators(output_directory, iglobal, plottable_operators, sl,
                            ground_truth_position, comm(iglobal), save_strategy)
            sl.save(join(output_directory, "pickle/") + _file_name_by_strategy(save_strategy, iglobal),
                    overwrite=overwrite)
            _save_random_state(output_directory, iglobal, save_strategy)

        callback(*((sl,) if _number_of_arguments(callback) == 1 else (sl, iglobal)))

        if mean.domain is not dom:
            raise RuntimeError
        if sl.domain is not dom:
            raise RuntimeError
    return (sl, mean) if return_final_position else sl


def _file_name(output_directory, name, index, prefix=""):
    op_direc = join(output_directory, name)
    return join(op_direc, f"{prefix}{index:03d}.png")


def _file_name_by_strategy(strategy, iglobal):
    if strategy == "all":
        return f"iteration_{iglobal}"
    elif strategy == "last":
        return "last"
    raise RuntimeError


def _save_random_state(output_directory, index, save_strategy):
    from ..random import getState
    file_name = join(output_directory, "pickle/nifty_random_state_")
    file_name += _file_name_by_strategy(save_strategy, index)
    with open(file_name, "wb") as f:
        f.write(getState())


def _plot_operators(output_directory, index, plottable_operators, sample_list, ground_truth, comm, save_strategy):
    if not isinstance(plottable_operators, dict):
        raise TypeError
    if not isdir(output_directory):
        raise RuntimeError(f"{output_directory} does not exist")
    if not isinstance(sample_list, SampleListBase):
        raise TypeError
    if ground_truth is not None and sample_list.domain != ground_truth.domain:
        raise TypeError

    for name, op in plottable_operators.items():
        gt = _op_force_or_none(op, ground_truth)
        _plot_samples(
            _file_name(output_directory, name, index, "samples_"), sample_list.iterator(op), gt, comm)
        if sample_list.n_samples() > 1:
            _plot_stats(
                _file_name(output_directory, name, index, "stats_"), *sample_list.sample_stat(op), gt, comm)

        op_direc = join(output_directory, name)
        if sample_list.n_samples() > 1:
            cfg = {"samples": True, "mean": True, "std": True}
        else:
            cfg = {"samples": True, "mean": False, "std": False}
        if name == "latent":
            continue
        if ground_truth is None or not _MPI_master(comm):
            ground_truth_sl = Nop()
        else:
            ground_truth_sl = SampleList([ground_truth])
        if h5py:
            file_name = join(op_direc, _file_name_by_strategy(save_strategy, index) + ".hdf5")
            sample_list.save_to_hdf5(file_name, op=op, overwrite=True, **cfg)
            file_name = join(op_direc, "ground_truth.hdf5")
            ground_truth_sl.save_to_hdf5(file_name, op=op, overwrite=True, samples=True)
        if astropy:
            try:
                file_name_base = join(op_direc, _file_name_by_strategy(save_strategy, index))
                sample_list.save_to_fits(file_name_base, op=op, overwrite=True, **cfg)
                file_name_base = join(op_direc, "ground_truth")
                ground_truth_sl.save_to_fits(file_name_base, op=op, overwrite=True, samples=True)
            except ValueError:
                pass


def _plot_samples(file_name, samples, ground_truth, comm):
    samples = list(samples)

    if _MPI_master(comm):
        if isinstance(samples[0].domain, DomainTuple):
            samples = [MultiField.from_dict({"": ss}) for ss in samples]
            if ground_truth is not None:
                ground_truth = MultiField.from_dict({"": ground_truth})
        if not all(isinstance(ss, MultiField) for ss in samples):
            raise TypeError
        keys = samples[0].keys()

        p = Plot()
        for kk in keys:
            try:
                plottable2D(samples[0][kk])
                twod = True
            except ValueError:
                twod = False

            single_samples = [ss[kk] for ss in samples]

            if twod:
                if ground_truth is not None:
                    p.add(ground_truth[kk], title=_append_key("Ground truth", kk))
                    p.add(None)
                for ii, ss in enumerate(single_samples):
                    if (ground_truth is None and ii == 16) or (ground_truth is not None and ii == 14):
                        break
                    p.add(ss, title=_append_key(f"Samples {ii}", kk))
            else:
                n = len(samples)
                alpha = n*[0.5]
                color = n*["maroon"]
                label = None
                if ground_truth is not None:
                    single_samples = [ground_truth] + single_samples
                    alpha = [1.] + alpha
                    color = ["green"] + color
                    label = ["Ground truth", "Samples"] + (n-1)*[None]
                p.add(samples, color=color, alpha=alpha, label=label, title=_append_key("", kk))
        p.output(name=file_name)


def _append_key(s, key):
    if key == "":
        return s
    return f"{s} ({key})"


def _plot_stats(file_name, mean, stddev, ground_truth, comm):
    p = Plot()
    if ground_truth is not None:
        p.add(ground_truth, title="Ground truth")
    p.add(mean, title="Mean")
    p.add(stddev, vmin=0, title="Standard deviation")
    if _MPI_master(comm):
        p.output(name=file_name, ny=2 if ground_truth is None else 3)


def _MPI_master(comm):
    return get_MPI_params_from_comm(comm)[2]


def _make_callable(obj):
    if callable(obj) and not (isinstance(obj, Minimizer) or isinstance(obj, IterationController) or
                              isinstance(obj, Operator)):
        return obj
    else:
        return lambda x: obj


def _op_force_or_none(operator, fld):
    if fld is None:
        return None
    return operator.force(fld)


def _want_metric(mini):
    from .descent_minimizers import L_BFGS, VL_BFGS, SteepestDescent

    # TODO Make this a property of the minimizer?
    if isinstance(mini, (SteepestDescent, L_BFGS, VL_BFGS)):
        return False
    return True

def _number_of_arguments(func):
    from inspect import signature
    return len(signature(func).parameters)
