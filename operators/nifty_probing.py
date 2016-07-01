# NIFTY (Numerical Information Field Theory) has been developed at the
# Max-Planck-Institute for Astrophysics.
#
# Copyright (C) 2015 Max-Planck-Society
#
# Author: Theo Steininger
# Project homepage: <http://www.mpa-garching.mpg.de/ift/nifty/>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from __future__ import division

from nifty.config import about
from nifty.space import Space
from nifty.field import Field
from nifty.nifty_utilities import direct_vdot


class prober(object):
    """
        ..                                    __        __
        ..                                  /  /      /__/
        ..      ______    _____   ______   /  /___    __   __ ___    ____ __
        ..    /   _   | /   __/ /   _   | /   _   | /  / /   _   | /   _   /
        ..   /  /_/  / /  /    /  /_/  / /  /_/  / /  / /  / /  / /  /_/  /
        ..  /   ____/ /__/     \______/  \______/ /__/ /__/ /__/  \____  /  class
        .. /__/                                                  /______/

        NIFTY class for probing (using multiprocessing)

        This is the base NIFTY probing class from which other probing classes
        (e.g. diagonal probing) are derived.

        When called, a probing class instance evaluates an operator or a
        function using random fields, whose components are random variables
        with mean 0 and variance 1. When an instance is called it returns the
        mean value of f(probe), where probe is a random field with mean 0 and
        variance 1. The mean is calculated as 1/N Sum[ f(probe_i) ].

        Parameters
        ----------
        op : operator
            The operator specified by `op` is the operator to be probed.
            If no operator is given, then probing will be done by applying
            `function` to the probes. (default: None)
        function : function, *optional*
            If no operator has been specified as `op`, then specification of
            `function` is non optional. This is the function, that is applied
            to the probes. (default: `op.times`)
        domain : space, *optional*
            If no operator has been specified as `op`, then specification of
            `domain` is non optional. This is the space that the probes live
            in. (default: `op.domain`)
        target : domain, *optional*
            `target` is the codomain of `domain`
            (default: `op.domain.get_codomain()`)
        random : string, *optional*
            the distribution from which the probes are drawn. `random` can be
            either "pm1" or "gau". "pm1" is a uniform distribution over {+1,-1}
            or {+1,+i,-1,-i}, respectively. "gau" is a normal distribution with
            zero-mean and unit-variance (default: "pm1")
        ncpu : int, *optional*
            the number of cpus to be used from parallel probing. (default: 2)
        nrun : int, *optional*
            the number of probes to be evaluated. If `nrun<ncpu`, it will be
            set to `ncpu`. (default: 8)
        nper : int, *optional*
            this number specifies how many probes will be evaluated by one
            worker. Afterwards a new worker will be created to evaluate a chunk
            of `nper` probes.
            If for example `nper=nrun/ncpu`, then every worker will be created
            for every cpu. This can lead to the case, that all workers but one
            are already finished, but have to wait for the last worker that
            might still have a considerable amount of evaluations left. This is
            obviously not very effective.
            If on the other hand `nper=1`, then for each evaluation a worker will
            be created. In this case all cpus will work until nrun probes have
            been evaluated.
            It is recommended to leave `nper` as the default value. (default: 1)
        var : bool, *optional*
            If `var` is True, then the variance of the sampled function will
            also be returned. The result is then a tuple with the mean in the
            zeroth entry and the variance in the first entry. (default: False)


        See Also
        --------
        diagonal_probing : A probing class to get the diagonal of an operator
        trace_probing : A probing class to get the trace of an operator


        Attributes
        ----------
        function : function
            the function, that is applied to the probes
        domain : space
            the space, where the probes live in
        target : space
            the codomain of `domain`
        random : string
            the random number generator used to create the probes
            (either "pm1" or "gau")
        ncpu : int
            the number of cpus used for probing
        nrun : int
            the number of probes to be evaluated, when the instance is called
        nper : int
            number of probes, that will be evaluated by one worker
        var : bool
            whether the variance will be additionally returned, when the
            instance is called
        quargs : dict
            Keyword arguments passed to `function` in each call.

    """
    def __init__(self, operator=None, function=None, domain=None,
                 codomain=None, random="pm1", nrun=8, varQ=False, **kwargs):
        """
        initializes a probing instance

        Parameters
        ----------
        op : operator
            The operator specified by `op` is the operator to be probed.
            If no operator is given, then probing will be done by applying
            `function` to the probes. (default: None)
        function : function, *optional*
            If no operator has been specified as `op`, then specification of
            `function` is non optional. This is the function, that is applied
            to the probes. (default: `op.times`)
        domain : space, *optional*
            If no operator has been specified as `op`, then specification of
            `domain` is non optional. This is the space that the probes live
            in. (default: `op.domain`)
        target : domain, *optional*
            `target` is the codomain of `domain`
            (default: `op.domain.get_codomain()`)
        random : string, *optional*
            the distribution from which the probes are drawn. `random` can be
            either "pm1" or "gau". "pm1" is a uniform distribution over {+1,-1}
            or {+1,+i,-1,-i}, respectively. "gau" is a normal distribution with
            zero-mean and unit-variance (default: "pm1")
        ncpu : int, *optional*
            the number of cpus to be used from parallel probing. (default: 2)
        nrun : int, *optional*
            the number of probes to be evaluated. If `nrun<ncpu`, it will be
            set to `ncpu`. (default: 8)
        nper : int, *optional*
            this number specifies how many probes will be evaluated by one
            worker. Afterwards a new worker will be created to evaluate a chunk
            of `nper` probes.
            If for example `nper=nrun/ncpu`, then every worker will be created
            for every cpu. This can lead to the case, that all workers but one
            are already finished, but have to wait for the last worker that
            might still have a considerable amount of evaluations left. This is
            obviously not very effective.
            If on the other hand `nper=1`, then for each evaluation a worker will
            be created. In this case all cpus will work until nrun probes have
            been evaluated.
            It is recommended to leave `nper` as the default value. (default: 1)
        var : bool, *optional*
            If `var` is True, then the variance of the sampled function will
            also be returned. The result is then a tuple with the mean in the
            zeroth entry and the variance in the first entry. (default: False)

        """
        # Case 1: no operator given. Check function and domain for general
        # sanity
        if operator is None:
            # check whether the given function callable
            if function is None or not hasattr(function, "__call__"):
                raise ValueError(about._errors.cstring(
                  "ERROR: invalid input: No function given or not callable."))
            # check given domain
            if domain is None or not isinstance(domain, Space):
                raise ValueError(about._errors.cstring(
                    "ERROR: invalid input: given domain is not a nifty space"))

        # Case 2: An operator is given. Take domain and function from that
        # if not given explicitly
        else:

            # Check 2.1 extract function
            # explicit function overrides operator function
            if function is None or not hasattr(function, "__call__"):
                try:
                    function = operator.times
                except(AttributeError):
                    raise ValueError(about._errors.cstring(
                        "ERROR: no explicit function given and given " +
                        "operator has no times method!"))

            # Check 2.2 check whether the given function is correctly bound to
            # the operator
            if operator != function.im_self:
                    raise ValueError(about._errors.cstring(
                        "ERROR: the given function is not a bound function " +
                        "of the operator!"))

            # Check 2.3 extract domain
            if domain is None or not isinstance(domain, Space):
                if (function in [operator.inverse_times,
                                 operator.adjoint_times]):
                    try:
                        domain = operator.target
                    except(AttributeError):
                        raise ValueError(about._errors.cstring(
                            "ERROR: no explicit domain given and given " +
                            "operator has no target!"))
                else:
                    try:
                        domain = operator.domain
                    except(AttributeError):
                        raise ValueError(about._errors.cstring(
                            "ERROR: no explicit domain given and given " +
                            "operator has no domain!"))

        self.function = function
        self.domain = domain

        # Check the given codomain
        if codomain is None:
            codomain = self.domain.get_codomain()
        else:
            assert(self.domain.check_codomain(codomain))

        self.codomain = codomain

        if random not in ["pm1", "gau"]:
            raise ValueError(about._errors.cstring(
                "ERROR: unsupported random key '" + str(random) + "'."))
        self.random = random

        # Parse the remaining arguments
        self.nrun = int(nrun)
        self.varQ = bool(varQ)
        self.kwargs = kwargs

    def configure(self, **kwargs):
        """
            changes the attributes of the instance

            Parameters
            ----------
            random : string, *optional*
                the random number generator used to create the probes (default: "pm1")
            ncpu : int, *optional*
                the number of cpus to be used for parallel probing. (default: 2)
            nrun : int, *optional*
                the number of probes to be evaluated. If `nrun<ncpu`, it will be
                set to `ncpu`. (default: 8)
            nper : int, *optional*
                number of probes, that will be evaluated by one worker (default: 1)
            var : bool, *optional*
                whether the variance will be additionally returned (default: False)

        """
        if "random" in kwargs:
            if kwargs.get("random") not in ["pm1", "gau"]:
                raise ValueError(about._errors.cstring(
                    "ERROR: unsupported random key '" +
                    str(kwargs.get("random")) + "'."))
            else:
                self.random = kwargs.get("random")

        if "nrun" in kwargs:
            self.nrun = int(kwargs.get("nrun"))

        if "varQ" in kwargs:
            self.varQ = bool(kwargs.get("varQ"))

    def generate_probe(self):
        """
            Generates a single probe

            Returns
            -------
            probe : field
                a random field living in `domain` with mean 0 and variance 1 in
                each component

        """
        return Field(self.domain,
                     codomain=self.codomain,
                     random=self.random)

    def evaluate_probe(self, probe, idnum=0):
        """
            Computes a single probing result given one probe

            Parameters
            ----------
            probe : field
                the field on which `function` will be applied
            idnum : int
                    the identification number of the probing

            Returns
            -------
            result : array-like
                the result of applying `function` to `probe`. The exact type
                depends on the function.

        """
        f = self.function(probe, **self.kwargs)
        return f

    def finalize(self, sum_of_probes, sum_of_squares, num):
        """
            Evaluates the probing results.

            Parameters
            ----------
            summa : numpy.array
                Sum of all probing results.
            num : int
                Number of successful probings (not returning ``None``).
            var : numpy.array
                Sum of all squared probing results

            Returns
            -------
            final : numpy.array
                The final probing result; 1/N Sum[ probing(probe_i) ]
            var : array-like
                The variance of the final probing result;
                1/(N(N-1)) Sum[( probing(probe_i) - final )^2];
                if the variance is returned, the return will be a tuple of
                (`final`,`var`).

        """
        # Check the success and efficiency of the probing
        if num < self.nrun:
            about.infos.cflush(
            " ( %u probe(s) failed, effectiveness == %.1f%% )\n"\
                %(self.nrun-num, 100*num/self.nrun))
            if num == 0:
                about.warnings.cprint("WARNING: probing failed.")
                return None
        else:
            about.infos.cflush("\n")

        if self.varQ:
            if num == 1:
                about.warnings.cprint(
                    "WARNING: Only one probe available -> infinite variance.")
                return (sum_of_probes, None)
            else:
                var = 1/(num-1)*(sum_of_squares - 1/num*(sum_of_probes**2))
                return (sum_of_probes*(1./num), var)
        else:
            return sum_of_probes*(1./num)

    def print_progress(self, num):  # > prints progress status upto 10 dots
        tenths = 1+(10*num//self.nrun)
        about.infos.cflush(("\b")*10+('.')*tenths+(' ')*(10-tenths))
    """
    def _single_probing(self,zipped): # > performs one probing operation
        # generate probe
        np.random.seed(zipped[0])
        probe = self.gen_probe()
        # do the actual probing
        return self.probing(zipped[1],probe)
    """

    def probe(self):  # > performs the probing operations one after another
        # initialize the variables
        sum_of_probes = 0
        sum_of_squares = 0
        num = 0

        for ii in xrange(self.nrun):
            print ('running probe ', ii)
            temp_probe = self.generate_probe()
            temp_result = self.evaluate_probe(probe=temp_probe)

            if temp_result is not None:
                sum_of_probes += temp_result
                if self.varQ:
                    sum_of_squares += ((temp_result).conjugate())*temp_result
                num += 1
                self.print_progress(num)
        about.infos.cflush(" done.")
        # evaluate
        return self.finalize(sum_of_probes, sum_of_squares, num)

    def __call__(self, loop=False, **kwargs):
        """

            Starts the probing process.
            All keyword arguments that can be given to `configure` can also be
            given to `__call__` and have the same effect.

            Parameters
            ----------
            loop : bool, *optional*
                if `loop` is True, then multiprocessing will be disabled and
                all probes are evaluated by a single worker (default: False)

            Returns
            -------
            results : see **Returns** in `evaluate`

            other parameters
            ----------------
            kwargs : see **Parameters** in `configure`

        """
        self.configure(**kwargs)
        return self.probe()

    def __repr__(self):
        return "<nifty_core.probing>"


class _specialized_prober(object):
    def __init__(self, operator, domain=None, inverseQ=False, **kwargs):
        # remove a potentially supplied function keyword argument
        try:
            kwargs.pop('function')
        except(KeyError):
            pass
        else:
            about.warnings.cprint(
                "WARNING: Dropped the supplied function keyword-argument!")

        if domain is None and not inverseQ:
            kwargs.update({'domain': operator.domain})
        elif domain is None and inverseQ:
            kwargs.update({'domain': operator.target})
        else:
            kwargs.update({'domain': domain})
        self.operator = operator

        self.prober = prober(function=self._probing_function,
                             **kwargs)

    def _probing_function(self, probe):
        return None

    def __call__(self, *args, **kwargs):
        return self.prober(*args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self.prober, attr)


class trace_prober(_specialized_prober):
    def __init__(self, operator, **kwargs):
        super(trace_prober, self).__init__(operator=operator,
                                           inverseQ=False,
                                           **kwargs)

    def _probing_function(self, probe):
        return direct_vdot(probe.conjugate(), self.operator.times(probe))


class inverse_trace_prober(_specialized_prober):
    def __init__(self, operator, **kwargs):
        super(inverse_trace_prober, self).__init__(operator=operator,
                                                   inverseQ=True,
                                                   **kwargs)

    def _probing_function(self, probe):
        return direct_vdot(probe.conjugate(),
                           self.operator.inverse_times(probe))


class diagonal_prober(_specialized_prober):
    def __init__(self, **kwargs):
        super(diagonal_prober, self).__init__(inverseQ=False,
                                              **kwargs)

    def _probing_function(self, probe):
        return probe.conjugate()*self.operator.times(probe)


class inverse_diagonal_prober(_specialized_prober):
    def __init__(self, operator, **kwargs):
        super(inverse_diagonal_prober, self).__init__(operator=operator,
                                                      inverseQ=True,
                                                      **kwargs)

    def _probing_function(self, probe):
        return probe.conjugate()*self.operator.inverse_times(probe)
