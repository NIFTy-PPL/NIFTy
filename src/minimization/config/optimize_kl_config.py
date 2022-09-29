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
# Copyright(C) 2022 Philipp Arras
# Author: Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import importlib
import os
from configparser import ConfigParser
from warnings import warn

from ...utilities import myassert

# FIXME point_estimates, constants?
# FIXME "2**lh0" looks weird. Change syntax? Use "$" for references?
# FIXME Cache operators (especially likelihoods)


class OptimizeKLConfig:
    """Plug config files into ift.optimize_kl.

    If you use this class for the first time, look at an example config file
    (e.g. `demos/getting_started_7_config_file.cfg`) before continuing to read
    the documentation.

    After parsing the config file the following steps are taken:

    - All `base` entries are interpreted. The options that are written out
      explicitly in a given section override what might be imported via a `base`
      entry. `base` inheritance does not work recursively (yet?).
    - Repetitions (e.g. "2*5,3*2") in `optimization.*` are expanded and replaced
      by an explicit list (e.g. "5,5,2,2,2"). The length of the resulting list
      is not allowed to be  longer than `total iterations`. If it is shorter,
      the last value is repeated.
    - All optimization stages (`optimization.*`) are joined into a single stage
      (`optimization.0`). For this, the quantity after the point `.` in
      `optimization.*` is interpreted as `int` and sorted afterwards. This means
      that for example `optimization.02` comes after `optimization.-1`. E.g.
      `optimization.1.1` is not allowed.

    For referring to e.g. likelihood energy operators, the star `*` is used as
    dereferencing operator. The value after the `*` refers to a section in the
    config file that must be instantiable via `OptimizeKLConfig.instantiate_section(key)`.

    Generally, all spaces (` `) in keys are internally replaced by underscores
    (`_`) such that they can be used in function calls.

    Parameters
    ----------
    config_parser : ConfigParser
        ConfigParser that contains all configuration.
    builders : dict
        Dictionary of functions that are used to instantiate e.g. operators.

    Example
    -------
    For an example of a typical config file, look at
    `demos/getting_started_7_config_file.cfg` in the nifty repository.

    Note
    ----
    Not treated by this class: export_operator_outputs, initial_position,
    initial_index, comm, inspect_callback, terminate_callback,
    return_final_position, resume

    Note
    ----
    Make sure that ConfigParser is case-sensitive by setting its attribute
    `optionxform` to `str` before parsing.
    """

    def __init__(self, config_parser, builders):
        if not isinstance(config_parser, ConfigParser):
            raise TypeError
        if config_parser.optionxform != str:
            warn("Consider setting `config_parser.optionxform = str`")
        self._cfg = config_parser
        self._builders = dict(builders)
        self._interpret_base()
        self._interpret_repetitions()
        self._join_optimization_stages()

    @classmethod
    def from_file(cls, file_name, builders):
        """
        Parameters
        ----------
        file_name : str
            File name of the config file that is imported.
        builders : dict
            Dictionary of functions that are used to instantiate e.g. operators.
        """
        cfg = ConfigParser()
        cfg.optionxform = str  # make keys case-sensitive
        if not os.path.isfile(file_name):
            raise RuntimeError(f"`{file_name}` not found")
        cfg.read(file_name)
        return cls(cfg, builders)

    def to_file(self, name):
        """Write configuration in standardized form to file.

        Parameters
        ----------
        name : str
            Path to which the config shall be written.
        """
        with open(name, "w") as f:
            self._cfg.write(f)

    def optimize_kl(self, **kwargs):
        """Do the inference and write the config file to output directory.

        All additional parameters to `ift.optimize_kl` can be passed via
        `kwargs`.
        """
        from ..optimize_kl import optimize_kl

        dct = dict(self)
        os.makedirs(dct["output_directory"], exist_ok=True)
        self.to_file(os.path.join(dct["output_directory"], "optimization.cfg"))
        return optimize_kl(**dct, **kwargs)

    def _interpret_base(self):
        """Replace `base` entry in every section by the content of the section it points to."""
        c = self._cfg
        for section in c:
            if "base" in c[section]:
                base_name = c[section]["base"]

                if base_name not in c:
                    raise RuntimeError(
                        f"the referred section `{base_name}` does not exist"
                    )
                if "base" in c[base_name]:
                    raise RuntimeError("recursive bases not allowed for now")

                # Replace base entry in section by the respective values
                c[section] = {**c[base_name], **c[section]}
                del c[section]["base"]

    def _interpret_repetitions(self):
        """Expand repretitions in sections of the form `optimization.*`.

        For example `2*NewtonCG` expands to `NewtonCG,NewtonCG`.

        If fewer entries than `total iterations` are present, fill up with the
        last value.
        """
        c = self._cfg

        # Only look at sections starting with "optimization."
        for optkey in filter(lambda x: x[:13] == "optimization.", c.keys()):
            sec = c[optkey]
            total_iterations = sec.getint("total iterations")
            for key in filter(lambda x: x != "total iterations", sec):
                if key == "base":
                    raise AssertionError(
                        "`base` must already be interpreted. This is a bug."
                    )

                # Expand multiply "*"
                if "," in sec[key]:
                    # Handle spaces around ","
                    entry_list_pre = map(lambda x: x.strip(), sec[key].split(","))
                    entry_list_post = []
                    for val in entry_list_pre:
                        # Nothing to expand because * not present or dereferencing operator
                        if "*" not in val or val[0] == "*":
                            entry_list_post.append(val)
                            continue

                        # Multiply "*" and dereferencing "*" mixed
                        splt = val.split("**")
                        if len(splt) == 2 and splt[0] != "" and splt[1] != "":
                            fac, val = splt
                            val = "*" + val
                            entry_list_post.extend(int(fac) * [val])
                            continue

                        # actual expansion
                        splt = val.split("*")
                        if len(splt) != 2:
                            raise RuntimeError(
                                f"the expression `{val}` cannot have more than one `*`"
                            )
                        fac, val = splt
                        entry_list_post.extend(int(fac) * [val])
                    sec[key] = ",".join(entry_list_post)

                # Fill up
                entry_list_pre = sec[key].split(",")
                diff = total_iterations - len(entry_list_pre)
                if diff < 0:
                    raise RuntimeError(
                        f"The number of total iterations ({total_iterations}) is at least {-diff} too small."
                    )
                entry_list_post = entry_list_pre + diff * [entry_list_pre[-1]]
                myassert(len(entry_list_post) == total_iterations)
                sec[key] = ",".join(entry_list_post)

    def _join_optimization_stages(self):
        """Join all optimization stages into one stage.

        All sections of the form `optimization.*` are combined into a single
        section called `optimization.0`.
        """
        c = self._cfg
        # Only look at sections starting with "optimization." But this time in ascending order
        lookup = {}
        for optkey in filter(lambda x: x[:13] == "optimization.", c.keys()):
            _, myid = optkey.split(".")
            lookup[int(myid)] = optkey
        optimization_keys = [lookup[kk] for kk in sorted(lookup)]
        # Sorting done.

        # Merge optimization sections together into "optimization.0"
        # Start with first one
        fst_key = optimization_keys[0]
        sec0 = c[fst_key]
        # Add the rest
        for optkey in optimization_keys[1:]:
            sec = c[optkey]

            for key in sec:
                if key == "total iterations":
                    sec0["total iterations"] = str(
                        sec0.getint("total iterations") + sec.getint("total iterations")
                    )
                    continue
                sec0[key] = ",".join([sec0[key], sec[key]])
            # has been merged into sec0 and can be deleted
            del c[optkey]

        # If user has chosen something different than optimization.0 as first stage, normalize it
        if fst_key != "optimization.0":
            tmp = c[fst_key]
            c["optimization.0"] = tmp
            del c[fst_key]

    def _to_callable(self, s, dtype=None):
        """Turn list separated by `,` into function that takes the index and returns the respective entry.

        Additionally all references indicated by `*` are instantiated.
        """

        def f(iteration):
            val = s.split(",")[iteration].strip()
            if val[0] == "*":  # is reference
                val = val[1:]
                val = self.instantiate_section(val)
            if val == "None":
                return None
            if dtype is not None:
                val = dtype(val)
            return val

        return f

    def instantiate_section(self, sec):
        """Instantiate object that is described by a section in the config file.

        There are two mechanisms how this instantiation works:

        - Look up the section key in the `self._builders` dictionary and call
          the respective function.
        - If `custom function` is specified in the section, pass all other
          entries of the section as arguments to the referred function.

        Before the instantiation is performed the inputs are transformed
        according to the type information that is passed in the config file. By
        default all values have type `str`. If `bool`, `float` or `int` shall be
        passed, the syntax `type :: value`, e.g. `float :: 1.2`, needs to be
        used in the config file.
        """
        dct = dict(self._cfg[sec])

        # Instantiate all references
        for kk in dct:
            val = dct[kk]
            if len(val) > 1 and val[0] == "*":  # is reference
                dct[kk] = self.instantiate_section(val[1:])

        # Replace all whitespaces with _
        # FIXME Is here the best place to do this?
        newdct = {}
        for kk, vv in dct.items():
            newdct[kk.replace(" ", "_")] = vv
        dct = newdct

        # Parse dtype
        for kk, vv in dct.items():
            if not isinstance(vv, str):
                continue
            tmp = tuple(map(lambda x: x.strip(), vv.split("::")))
            if len(tmp) == 2:  # type information available
                if tmp[0] == "bool":
                    if tmp[1].lower() == "true":
                        vv = True
                    elif tmp[1].lower() == "false":
                        vv = False
                    else:
                        ValueError(f"{tmp[1]} is not boolean")
                elif tmp[0] == "float":
                    vv = float(tmp[1])
                elif tmp[0] == "int":
                    vv = int(tmp[1])
                elif tmp[0] == "None":
                    vv = None
            dct[kk] = vv

        # Plug into builder or something else
        if sec in self._builders:
            return self._builders[sec](**dct)
        if "custom_function" in dct:
            mod_name, func_name = dct.pop("custom_function").rsplit(".", 1)
            mod = importlib.import_module(mod_name)
            func = getattr(mod, func_name)
            return func(**dct)
        raise RuntimeError(
            f"Provide build routine for `{sec}` in builders dictionary or "
            "reference a `custom_function` in the config file."
        )

    def __iter__(self):
        """Enable conversion to `dict` such that the result of this class can
        easily be passed into `ift.optimize_kl`."""
        cdyn = self._cfg["optimization.0"]

        # static
        copt = self._cfg["optimization"]
        yield "output_directory", copt["output directory"]
        yield "save_strategy", copt["save strategy"]
        yield "plot_energy_history", True
        yield "plot_minisanity_history", True

        # dynamic
        yield "total_iterations", int(cdyn["total iterations"])
        for key in [
            "likelihood_energy",
            "n_samples",
            "transitions",
            "kl_minimizer",
            "sampling_iteration_controller",
            "nonlinear_sampling_minimizer",
        ]:
            key1 = key.replace("_", " ")
            if key == "n_samples":
                yield key, self._to_callable(cdyn[key1], int)
            else:
                yield key, self._to_callable(cdyn[key1])

    def __str__(self):
        s = []
        for key, val in self._cfg.items():
            s += [key]
            s += [f"  {kk}: {vv}" for kk, vv in val.items()]
            s += [""]
        return "\n".join(s)

    def __eq__(self, other):
        for a in "_cfg", "_builders":
            if getattr(self, a) != getattr(other, a):
                return False
        return True
