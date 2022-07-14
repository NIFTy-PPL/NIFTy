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
# Copyright(C) 2022 Max-Planck-Society, Philipp Arras
# Author: Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import pickle
from functools import reduce
from os import makedirs
from os.path import isdir, isfile, join
from warnings import warn

import numpy as np

from ..domain_tuple import DomainTuple
from ..multi_domain import MultiDomain
from ..multi_field import MultiField
from ..operators.counting_operator import CountingOperator
from ..operators.energy_operators import StandardHamiltonian
from ..operators.operator import Operator
from ..operators.scaling_operator import ScalingOperator
from ..plot import Plot, plottable2D
from ..sugar import from_random
from ..utilities import (Nop, check_MPI_equality,
                         check_MPI_synced_random_state, check_object_identity,
                         get_MPI_params_from_comm)
from .energy_adapter import EnergyAdapter
from .iteration_controllers import EnergyHistory, IterationController
from .kl_energies import SampledKLEnergy
from .minimizer import Minimizer
from .sample_list import (ResidualSampleList, SampleList, SampleListBase,
                          _barrier)

try:
    import h5py
except ImportError:
    h5py = False

try:
    import astropy
except ImportError:
    astropy = False

_output_directory = None
_save_strategy = None


def optimize_kl(likelihood_energy,
                total_iterations,
                n_samples,
                kl_minimizer,
                sampling_iteration_controller,
                nonlinear_sampling_minimizer,
                constants=[],
                point_estimates=[],
                transitions=None,
                export_operator_outputs={},
                output_directory=None,
                initial_position=None,
                initial_index=0,
                comm=None,
                inspect_callback=None,
                terminate_callback=None,
                plot_energy_history=True,
                plot_minisanity_history=True,
                save_strategy="last",
                return_final_position=False,
                resume=False,
                sanity_checks=True):
    """Provide potentially useful interface for standard KL minimization.

    The parameters `likelihood_energy`, `kl_minimizer`,
    `sampling_iteration_controller`, `nonlinear_sampling_minimizer`, `constants`,
    `point_estimates` and `comm` also accept a function as input that takes the
    index of the global iteration as input and returns the respective value that
    should be used for that iteration.

    High-level pseudo code of the algorithm, that is implemented by `optimize_kl`:

    .. code-block:: none

        for ii in range(initial_index, total_iterations):
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
        standard normal distributed. It needs to have a MultiDomain as Domain.
    total_iterations : int
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
    transitions : Operator or callable
        Operator that is applied at the beginning of each iteration to the
        latent position. This can be used if parts of the likelihood_energy are
        replaced and thereby have a different domain. If None is passed, no
        transition will be applied. Default is None.
    export_operator_outputs : dict
        Dictionary of operators that are exported during the minimization. The
        key contains a string that serves as identifier. The value of the
        dictionary is an operator.
    output_directory : str or None
        Directory in which all output files are saved. If None, no output is
        stored.  Default: None.
    initial_position : :class:`nifty8.field.Field`, :class:`nifty8.multi_field.MultiField` or None
        Position in the definition space of `likelihood_energy` from which the
        optimization is started. If `None`, it starts at a random, normal
        distributed position with standard deviation 0.1. Default: None.
    initial_index : int
        Initial index that is used to enumerate the output files. May be used
        if `optimize_kl` is called multiple times. Default: 0.
    comm : MPI communicator or None
        MPI communicator for distributing samples over MPI tasks. If `None`,
        the samples are not distributed. Default: None.
    inspect_callback : callable or None
        Function that is called after every global iteration. It can be either a
        function with one argument (then the latest sample list is passed), a
        function with two arguments (in which case the latest sample list and
        the global iteration index are passed). Default: None.
    terminate_callback : callable or None
        Function that is called after every global iteration and after
        `inspect_callback` if present.  It can be either None or a function
        that takes the global iteration index as input and returns a boolean.
        If the return value is true, the global loop in `optimize_kl` is
        terminated. Default: None.
    plot_energy_history : bool
        Determine if the KLEnergy values shall be plotted or not. Default: True.
    plot_minisanity_history : bool
        Determine if the reduced chi-square values computed by minisanity shall
        be plotted or not. Default: True.
    save_strategy : str
        If "last", only the samples of the last global iteration are stored. If
        "all", all intermediate samples are written to disk. `save_strategy` is
        only applicable if `output_directory` is not None. Default: "last".
    return_final_position : bool
        Determine if the final position of the minimization shall be return.
        May be useful to feed it as `initial_position` into another
        `optimize_kl` call. Default: False.
    resume : bool
        Resume partially run optimization. If `True` and `output_directory`
        contains `last_finished_iteration`, `initial_index` and
        `initial_position` are ignored and read from the output directory
        instead. If `last_finished_iteration` is not a file, the value of
        `initial_position` is used instead. Default: False.
    sanity_checks : bool
        Check if all domains of likelihoods and transitions fit together. This
        is potentially expensive because all likelihoods have to be instantiated
        multiple times. Default: True.

    Returns
    -------
    sl : SampleList

    mean : :class:`nifty8.field.Field` or :class:`nifty8.multi_field.MultiField` (optional)

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

    if not isinstance(export_operator_outputs, dict):
        raise TypeError
    if len(set(["pickle"]) & set(export_operator_outputs.keys())) != 0:
        raise ValueError("The key `pickle` in `export_operator_outputs` is reserved.")
    if not isinstance(initial_index, int):
        raise TypeError
    if save_strategy not in ["all", "last"]:
        raise ValueError(f"Save strategy '{save_strategy}' not supported.")
    if output_directory is None and resume:
        raise ValueError("Can only resume minimization if output_directory is not None")

    likelihood_energy = _make_callable(likelihood_energy)
    kl_minimizer = _make_callable(kl_minimizer)
    sampling_iteration_controller = _make_callable(sampling_iteration_controller)
    nonlinear_sampling_minimizer = _make_callable(nonlinear_sampling_minimizer)
    constants = _make_callable(constants)
    point_estimates = _make_callable(point_estimates)
    transitions = _make_callable(transitions)
    n_samples = _make_callable(n_samples)
    comm = _make_callable(comm)
    if inspect_callback is None:
        inspect_callback = lambda x: None
    if terminate_callback is None:
        terminate_callback = lambda x: False

    if output_directory is not None:
        global _output_directory
        global _save_strategy
        _output_directory = output_directory
        _save_strategy = save_strategy

        lfile = join(output_directory, "last_finished_iteration")
        if resume and isfile(lfile):
            with open(lfile) as f:
                last_finished_index = int(f.read())
            initial_index = last_finished_index + 1
            fname = _file_name_by_strategy(last_finished_index)
            fname = reduce(join, [output_directory, "pickle", fname])
            if isfile(fname + ".mean.pickle"):
                initial_position = ResidualSampleList.load_mean(fname)
            else:
                sl = SampleList.load(fname)
                myassert(sl.n_samples == 1)
                initial_position = sl.local_item(0)
            _load_random_state(last_finished_index)
            energy_history = _pickle_load_values(last_finished_index, 'energy_history')

            if initial_index == total_iterations:
                if isfile(fname + ".mean.pickle"):
                    sl = ResidualSampleList.load(fname)
                return (sl, initial_position) if return_final_position else sl

    if sanity_checks:
        if initial_index >= total_iterations:
            raise ValueError("Initial index is bigger than total iterations: "
                            f"{initial_index} >= {total_iterations}")

        for iglobal in range(initial_index, total_iterations):
            for (obj, cls) in [(likelihood_energy, Operator), (kl_minimizer, DescentMinimizer),
                            (nonlinear_sampling_minimizer, (DescentMinimizer, type(None))),
                            (constants, (list, tuple)), (point_estimates, (list, tuple)),
                            (transitions, (Operator, type(None))), (n_samples, int)]:
                if not isinstance(obj(iglobal), cls):
                    raise TypeError(f"{obj(iglobal)} is not instance of {cls} but rather {type(obj(iglobal))}")

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
        myassert(_number_of_arguments(inspect_callback) in [1, 2])
        myassert(_number_of_arguments(terminate_callback) == 1)

        if not likelihood_energy(initial_index).target is DomainTuple.scalar_domain():
            raise TypeError

    # Initial position
    check_MPI_synced_random_state(comm(initial_index))
    if initial_position is None:
        from ..sugar import full, makeDomain
        mean = full(makeDomain({}), 0.)
    else:
        mean = initial_position
    # /Initial position

    # Automatic transitions
    def trans(iglobal, prev_dom, next_dom):
        res = transitions(iglobal)
        if res is not None:
            return res

        if iglobal == 0:
            dom = mean.domain
        else:
            dom = prev_dom
        tgt = next_dom

        if dom is tgt:
            return Operator.identity_operator(dom)
        if set(dom.keys()).issubset(set(tgt.keys())):
            return _InitializeOperator(dom, tgt, std=0.1)
        raise RuntimeError

    if sanity_checks:
        for iglobal in range(initial_index, total_iterations):
            next_dom = likelihood_energy(iglobal).domain
            if iglobal == 0:
                prev_dom = mean.domain
            else:
                prev_dom = likelihood_energy(iglobal-1).domain

            next_dom1 = trans(iglobal, prev_dom, next_dom).target
            if next_dom != next_dom1:
                raise RuntimeError(f"The domain of lh energy #{iglobal} should equal the target of transition.\n"
                                f"{next_dom}\n\n"
                                f"{next_dom1}")
            myassert(prev_dom is trans(iglobal, prev_dom, next_dom).domain)
    # /Automatic transitions

    if output_directory is not None:
        # Create all necessary subfolders
        if _MPI_master(comm(initial_index)):
            makedirs(output_directory, exist_ok=True)
            subfolders = ["pickle"] + list(export_operator_outputs.keys())
            if plot_energy_history:
                subfolders += ["energy_history"]
            if plot_minisanity_history:
                subfolders += ["minisanity_history"]
            for subfolder in subfolders:
                makedirs(join(output_directory, subfolder), exist_ok=True)

    if initial_index == 0:
        energy_history = EnergyHistory()

    for iglobal in range(initial_index, total_iterations):
        lh = likelihood_energy(iglobal)
        mean = trans(iglobal, mean.domain, lh.domain)(mean)
        dom = mean.domain
        mf_dom = isinstance(dom, MultiDomain)
        if not mf_dom:
            warn("In the upcoming release nifty8.optimize_kl will no longer support "
                 "DomainTuple as latent position. You can use "
                 "`likelihood_energy.ducktape(key)` to convert a DomainTuple domain into "
                 "a MultiDomain. Please report at `c@philipp-arras.de` if you use this "
                 "feature and would like to see it continued.", DeprecationWarning)

        count = CountingOperator(lh.domain)
        ham = StandardHamiltonian(lh @ count, sampling_iteration_controller(iglobal))
        minimizer = kl_minimizer(iglobal)
        mean_iter = mean.extract(ham.domain)

        # TODO Distributing the domain of the likelihood is not supported (yet)
        check_MPI_synced_random_state(comm(iglobal))
        check_MPI_equality(lh.domain, comm(iglobal))
        check_MPI_equality(mean.domain, comm(iglobal))
        check_MPI_equality(mean, comm(iglobal))

        if n_samples(iglobal) == 0:
            e = EnergyAdapter(mean_iter, ham, constants=constants(iglobal),
                              want_metric=_want_metric(minimizer))
            if comm(iglobal) is None:
                e, _ = minimizer(e)
                mean = MultiField.union([mean, e.position]) if mf_dom else e.position
                sl = SampleList([mean])
                energy_history.append((iglobal, e.value))
            else:
                warn("Have detected MPI communicator for optimizing Hamiltonian. Will use only "
                     "the rank0 task and communicate the result afterwards to all other tasks.")
                if _MPI_master(comm(iglobal)):
                    e, _ = minimizer(e)
                    mean = MultiField.union([mean, e.position]) if mf_dom else e.position
                    sl = SampleList([mean], comm=comm(iglobal), domain=dom)
                    energy_history.append((iglobal, e.value))
                else:
                    mean = None
                    sl = SampleList([], comm=comm(iglobal), domain=dom)
                _barrier(comm(iglobal))
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
            sl = e.samples.at(mean)
            energy_history.append((iglobal, e.value))

        if output_directory is not None:
            _export_operators(iglobal, export_operator_outputs, sl, comm(iglobal))
            sl.save(join(output_directory, "pickle/") + _file_name_by_strategy(iglobal),
                    overwrite=True)
            _save_random_state(iglobal)

            if _MPI_master(comm(iglobal)):
                with open(join(output_directory, "last_finished_iteration"), "w") as f:
                    f.write(str(iglobal))
                _pickle_save_values(iglobal, 'energy_history', energy_history)
                if plot_energy_history:
                    _plot_energy_history(iglobal, energy_history)
        _barrier(comm(iglobal))

        _minisanity(lh, iglobal, sl, comm, plot_minisanity_history)
        _barrier(comm(iglobal))

        _counting_report(count, iglobal, comm)

        _handle_inspect_callback(inspect_callback, sl, iglobal)
        _barrier(comm(iglobal))

        if _handle_terminate_callback(terminate_callback, iglobal, comm):
            break
        _barrier(comm(iglobal))

        del lh

    return (sl, mean) if return_final_position else sl


def _file_name(name, index, prefix=""):
    op_direc = join(_output_directory, name)
    return join(op_direc, f"{prefix}{index:03d}.png")


def _file_name_by_strategy(iglobal, save_strategy='global_strategy'):
    if save_strategy == 'global_strategy':
        save_strategy = _save_strategy
    if save_strategy == "all":
        return f"iteration_{iglobal}"
    elif save_strategy == "last":
        return "last"
    raise RuntimeError


def _save_random_state(index):
    from ..random import getState
    file_name = join(_output_directory, "pickle/nifty_random_state_")
    file_name += _file_name_by_strategy(index)
    with open(file_name, "wb") as f:
        f.write(getState())


def _load_random_state(index):
    from ..random import setState
    file_name = join(_output_directory, "pickle/nifty_random_state_")
    file_name += _file_name_by_strategy(index)
    with open(file_name, "rb") as f:
        setState(f.read())


def _pickle_save_values(index, name, val):
    file_name = join(_output_directory, f"pickle/{name}_")
    file_name += _file_name_by_strategy(index)
    with open(file_name, "wb") as f:
        pickle.dump(val, f)


def _pickle_load_values(index, name):
    file_name = join(_output_directory, f"pickle/{name}_")
    file_name += _file_name_by_strategy(index)
    with open(file_name, "rb") as f:
        val = pickle.load(f)
    return val


def _export_operators(index, export_operator_outputs, sample_list, comm):
    if not isinstance(export_operator_outputs, dict):
        raise TypeError
    if not isdir(_output_directory):
        raise RuntimeError(f"{_output_directory} does not exist")
    if not isinstance(sample_list, SampleListBase):
        raise TypeError

    for name, op in export_operator_outputs.items():
        if not _is_subdomain(op.domain, sample_list.domain):
            continue

        op_direc = join(_output_directory, name)
        if sample_list.n_samples > 1:
            cfg = {"samples": True, "mean": True, "std": True}
        else:
            cfg = {"samples": True, "mean": False, "std": False}

        if h5py:
            file_name = join(op_direc, _file_name_by_strategy(index) + ".hdf5")
            sample_list.save_to_hdf5(file_name, op=op, overwrite=True, **cfg)

        if astropy:
            try:
                file_name_base = join(op_direc, _file_name_by_strategy(index))
                sample_list.save_to_fits(file_name_base, op=op, overwrite=True, **cfg)
            except ValueError:
                pass


def _plot_energy_history(index, energy_history):
    import matplotlib.pyplot as plt
    fname = join(_output_directory, 'energy_history',
                 '{}_' + _file_name_by_strategy(index) + '.png')

    E = np.array(energy_history.energy_values)

    # energy value plot
    p = Plot()
    p.add(energy_history, skip_timestamp_conversion=True, xlabel='iteration',
          ylabel=r'E', yscale='log' if (E > 0.).all() else 'linear')
    p.output(title='energy history', name=fname.format('energy_history'))

    # energy change plot
    if index > 0:
        ts = np.array(energy_history.time_stamps[1:])
        dE = E[1:] - E[:-1]
        idx_pos = (dE > 0)
        idx_neg = (dE < 0)
        plt.plot(ts[idx_pos], dE[idx_pos], '^', color='red', label='positive')
        plt.plot(ts[idx_neg], -dE[idx_neg], 'v', color='green', label='negative')
        plt.yscale('log')
        plt.title('energy change w.r.t. previous step')
        plt.xlabel('iteration')
        plt.ylabel(r'$\Delta\,$E')
        plt.legend()
        plt.savefig(fname.format('energy_change_history'))
        plt.clf()


def _append_key(s, key):
    if key == "":
        return s
    return f"{s} ({key})"


def _minisanity(likelihood_energy, iglobal, sl, comm, plot_minisanity_history):
    from ..extra import minisanity

    s, ms_val = minisanity(likelihood_energy, sl, terminal_colors=False,
                           return_values=True)
    check_MPI_equality(ms_val, comm(iglobal))
    _report_to_logger_and_file(s, "minisanity.txt", iglobal, comm, True, True,
                               True)

    if _MPI_master(comm(iglobal)) and _output_directory is not None:
        # load/create minisanity_history object
        value_type_keys = ['redchisq', 'scmean']
        category_keys = ['data_residuals', 'latent_variables']

        if iglobal == 0:
            mh = {tk: {ck: {} for ck in category_keys} for tk in value_type_keys}
        else:
            mh = _pickle_load_values(iglobal - 1, 'minisanity_history')

        key_set_mh = {}
        key_set_msval = {}
        for ck in category_keys:
            key_set_mh[ck] = set(mh['redchisq'][ck].keys())
            key_set_msval[ck] = set(ms_val['redchisq'][ck].keys())

        for tk in value_type_keys:
            for ck in category_keys:
                # all keys not yet in minisanity history
                for ek in key_set_msval[ck] - key_set_mh[ck]:
                    v = ms_val[tk][ck][ek]
                    mh[tk][ck][ek] = {}
                    mh[tk][ck][ek]['index'] = [iglobal, ]
                    mh[tk][ck][ek]['mean'] = [v['mean'], ]
                    mh[tk][ck][ek]['std'] = [v['std'], ]
                # all keys already present in minisanity history
                for ek in key_set_msval[ck] & key_set_mh[ck]:
                    v = ms_val[tk][ck][ek]
                    mh[tk][ck][ek]['index'].append(iglobal)
                    mh[tk][ck][ek]['mean'].append(v['mean'])
                    mh[tk][ck][ek]['std'].append(v['std'])

        _pickle_save_values(iglobal, 'minisanity_history', mh)

        if plot_minisanity_history:
            _plot_minisanity_history(iglobal, mh)


def _plot_minisanity_history(index, minisanity_history):
    import matplotlib.pyplot as plt
    from matplotlib.cm import plasma

    mhrcs = minisanity_history['redchisq']

    labels = []
    vals = []
    idxs = []

    n_dr = 0
    for kk in mhrcs['data_residuals'].keys():
        labels.append(f'residuals: {n_dr}')
        vals.append(mhrcs['data_residuals'][kk]['mean'])
        idxs.append(mhrcs['data_residuals'][kk]['index'])
        n_dr += 1

    n_lv = 0
    for kk in mhrcs['latent_variables'].keys():
        labels.append(f'latent: {kk}')
        vals.append(mhrcs['latent_variables'][kk]['mean'])
        idxs.append(mhrcs['latent_variables'][kk]['index'])
        n_lv += 1

    n_tot = n_dr + n_lv

    colors = [plasma(x) for x in np.linspace(0, 0.95, n_tot)]

    linestyles = ['-'] * n_tot
    for ii in range(1, n_tot, 2):
        linestyles[ii] = '--'

    vals = [np.array(v) for v in vals]
    idxs = [np.array(i) for i in idxs]

    plt.figure()
    for i in range(n_tot):
        plt.plot(idxs[i], vals[i], label=labels[i], color=colors[i], marker='.',
                 linestyle=linestyles[i])

    xlim = plt.xlim()
    plt.hlines(1., xlim[0], xlim[1], linestyle='dashed', color='black', linewidth=2, zorder=-1)
    plt.xlim(*xlim)
    plt.yscale('log')

    plt.title(r'reduced $\chi^2$ values')
    plt.xlabel('iteration')
    plt.ylabel(r'red. $\chi^2$')
    plt.legend(loc='upper right')
    plt.savefig(join(_output_directory, 'minisanity_history',
                     'minisanity_history_' + _file_name_by_strategy(index) + '.png'))
    plt.clf()


def _counting_report(count, iglobal, comm):
    _report_to_logger_and_file(count.report(), "counting_report.txt", iglobal,
                               comm, _output_directory is None, True, False)


def _report_to_logger_and_file(report, file_name, iglobal, comm, to_logger,
                               to_file, only_master):
    from datetime import datetime

    from ..logger import logger
    from ..utilities import allreduce_sum

    intro = f"Finished index: {iglobal}\nCurrent datetime: {datetime.now()}\n"

    if not only_master:
        report = allreduce_sum([[report]], comm(iglobal))
        report = [f"Task {ii}\n{rr}" for ii, rr in enumerate(report)]
        report = "\n".join(report)

    if _MPI_master(comm(iglobal)):
        if to_logger:
            logger.info(report)
        if _output_directory is not None and to_file:
            with open(join(_output_directory, file_name), "a", encoding="utf-8") as f:
                f.write(intro + report + "\n\n")


def _handle_inspect_callback(inspect_callback, sl, iglobal):
    if _number_of_arguments(inspect_callback) == 1:
        inp = (sl, )
    elif _number_of_arguments(inspect_callback) == 2:
        inp = (sl, iglobal)
    inspect_callback(*inp)


def _handle_terminate_callback(terminate_callback, iglobal, comm):
    terminate = terminate_callback(iglobal)
    check_MPI_equality(terminate, comm(iglobal))
    return terminate


def _MPI_master(comm):
    return get_MPI_params_from_comm(comm)[2]


def _make_callable(obj):
    if callable(obj) and not (isinstance(obj, Minimizer) or isinstance(obj, IterationController) or
                              isinstance(obj, Operator)):
        return obj
    else:
        return lambda x: obj


def _want_metric(mini):
    from .descent_minimizers import L_BFGS, VL_BFGS, SteepestDescent

    # TODO Make this a property of the minimizer?
    if isinstance(mini, (SteepestDescent, L_BFGS, VL_BFGS)):
        return False
    return True


def _number_of_arguments(func):
    from inspect import signature
    return len(signature(func).parameters)


def _is_subdomain(sub_domain, total_domain):
    if not isinstance(sub_domain, (MultiDomain, DomainTuple)):
        raise TypeError
    if isinstance(sub_domain, DomainTuple):
        return sub_domain == total_domain
    return all(kk in total_domain.keys() and vv == total_domain[kk]
               for kk, vv in sub_domain.items())


class _InitializeOperator(Operator):
    def __init__(self, domain, target, std):
        self._domain = MultiDomain.make(domain)
        self._target = MultiDomain.make(target)
        self._has_been_called = False
        self._std = float(std)

    def apply(self, x):
        from ..sugar import from_random, is_linearization, makeDomain
        from ..utilities import myassert

        self._check_input(x)
        if self._has_been_called:
            raise RuntimeError("This operator can only be called once since it returns random numbers.")
        self._has_been_called = True
        if is_linearization(x):
            raise TypeError("Only MultiFields are supported as input")
        diff_keys = set(self._target.keys()) - set(self._domain.keys())
        diff_domain = makeDomain({kk: self._target[kk] for kk in diff_keys})
        diff_fld = from_random(diff_domain, std=self._std)
        myassert(len(set(diff_fld.keys()) & set(self._domain.keys())) == 0)
        return x.unite(diff_fld)
