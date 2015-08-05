## NIFTY (Numerical Information Field Theory) has been developed at the
## Max-Planck-Institute for Astrophysics.
##
## Copyright (C) 2015 Max-Planck-Society
##
## Author: Marco Selig
## Project homepage: <http://www.mpa-garching.mpg.de/ift/nifty/>
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
## See the GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program. If not, see <http://www.gnu.org/licenses/>.

from __future__ import division

import os
from sys import stdout as so
import numpy as np
from multiprocessing import Pool as mp
from multiprocessing import Value as mv
from multiprocessing import Array as ma

from nifty.nifty_about import about
from nifty.nifty_core import space, \
                         field 



##-----------------------------------------------------------------------------

class _share(object):

    __init__ = None

    @staticmethod
    def _init_share(_sum,_num,_var):
        _share.sum = _sum
        _share.num = _num
        _share.var = _var

##-----------------------------------------------------------------------------

##=============================================================================

class probing(object):
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
    def __init__(self,op=None,function=None,domain=None,target=None,random="pm1",ncpu=2,nrun=8,nper=1,var=False,**quargs):
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
        if(op is None):
            ## check whether callable
            if(function is None)or(not hasattr(function,"__call__")):
                raise TypeError(about._errors.cstring("ERROR: invalid input."))
            ## check given domain
            if(domain is None)or(not isinstance(domain,space)):
                raise TypeError(about._errors.cstring("ERROR: invalid input."))
        else:
            from nifty_operators import operator
            if(not isinstance(op,operator)):
                raise TypeError(about._errors.cstring("ERROR: invalid input."))
            ## check whether callable
            if(function is None)or(not hasattr(function,"__call__")):
                function = op.times
            elif(op==function):
                function = op.times
            ## check whether correctly bound
            if(op!=function.im_self):
                raise NameError(about._errors.cstring("ERROR: invalid input."))
            ## check given shape and domain
            if(domain is None)or(not isinstance(domain,space)):
                if(function in [op.inverse_times,op.adjoint_times]):
                    domain = op.target
                else:
                    domain = op.domain
            else:
                if(function in [op.inverse_times,op.adjoint_times]):
                    op.target.check_codomain(domain) ## a bit pointless
                    if(target is None)or(not isinstance(target,space)):
                        target = op.target
                else:
                    op.domain.check_codomain(domain) ## a bit pointless
                    if(target is None)or(not isinstance(target,space)):
                        target = op.domain

        self.function = function
        self.domain = domain

        ## check codomain
        if(target is None):
            target = self.domain.get_codomain()
        else:
            self.domain.check_codomain(target) ## a bit pointless
        self.target = target

        if(random not in ["pm1","gau"]):
            raise ValueError(about._errors.cstring("ERROR: unsupported random key '"+str(random)+"'."))
        self.random = random

        self.ncpu = int(max(1,ncpu))
        self.nrun = int(max(self.ncpu,nrun))
        if(nper is None):
            self.nper = None
        else:
            self.nper = int(max(1,min(self.nrun//self.ncpu,nper)))

        self.var = bool(var)

        self.quargs = quargs

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def configure(self,**kwargs):
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
        if("random" in kwargs):
            if(kwargs.get("random") not in ["pm1","gau"]):
                raise ValueError(about._errors.cstring("ERROR: unsupported random key '"+str(kwargs.get("random"))+"'."))
            self.random = kwargs.get("random")

        if("ncpu" in kwargs):
            self.ncpu = int(max(1,kwargs.get("ncpu")))
        if("nrun" in kwargs):
            self.nrun = int(max(self.ncpu,kwargs.get("nrun")))
        if("nper" in kwargs):
            if(kwargs.get("nper") is None):
                self.nper = None
            else:
                self.nper = int(max(1,min(self.nrun//self.ncpu,kwargs.get("nper"))))

        if("var" in kwargs):
            self.var = bool(kwargs.get("var"))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def gen_probe(self):
        """
            Generates a single probe

            Returns
            -------
            probe : field
                a random field living in `domain` with mean 0 and variance 1 in
                each component

        """
        return field(self.domain,val=None,target=self.target,random=self.random)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def probing(self,idnum,probe):
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
        f = self.function(probe,**self.quargs)
        if(isinstance(f,field)):
            return f.val
        else:
            return f

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def evaluate(self,summa,num,var):
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
        if(num<self.nrun):
            about.infos.cflush(" ( %u probe(s) failed, effectiveness == %.1f%% )\n"%(self.nrun-num,100*num/self.nrun))
            if(num==0):
                about.warnings.cprint("WARNING: probing failed.")
                return None
        else:
            about.infos.cflush("\n")

        if(summa.size==1):
            summa = summa.flatten(order='C')[0]
            if(self.var):
                var = var.flatten(order='C')[0]
        if(np.iscomplexobj(summa))and(np.all(np.imag(summa)==0)):
            summa = np.real(summa)

        final = summa*(1/num)
        if(self.var):
            if(num==1):
                about.warnings.cprint("WARNING: infinite variance.")
                return final,None
            else:
                var = var*(1/(num*(num-1)))-np.real(np.conjugate(final)*final)*(1/(num-1))
                return final,var
        else:
            return final

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _progress(self,irun): ## > prints progress status by in upto 10 dots
        tenths = 1+(10*irun//self.nrun)
        about.infos.cflush(("\b")*10+('.')*tenths+(' ')*(10-tenths))

    def _single_probing(self,zipped): ## > performs one probing operation
        ## generate probe
        np.random.seed(zipped[0])
        probe = self.gen_probe()
        ## do the actual probing
        return self.probing(zipped[1],probe)

    def _serial_probing(self,zipped): ## > performs the probing operation serially
        try:
            result = self._single_probing(zipped)
        except Exception as exception:
            raise exception
        except BaseException: ## capture system-exiting exception including KeyboardInterrupt
            raise Exception(about._errors.cstring("ERROR: unknown."))
        else:
            if(result is not None):
                result = np.array(result).flatten(order='C')
                if(np.iscomplexobj(result)):
                    _share.sum[0].acquire(block=True,timeout=None)
                    _share.sum[0][:] += np.real(result)
                    _share.sum[0].release()
                    _share.sum[1].acquire(block=True,timeout=None)
                    _share.sum[1][:] += np.imag(result)
                    _share.sum[1].release()
                else:
                    _share.sum.acquire(block=True,timeout=None)
                    _share.sum[:] += result
                    _share.sum.release()
                _share.num.acquire(block=True,timeout=None)
                _share.num.value += 1
                _share.num.release()
                if(self.var):
                    _share.var.acquire(block=True,timeout=None)
                    _share.var[:] += np.real(np.conjugate(result)*result)
                    _share.var.release()
                self._progress(_share.num.value)

    def _parallel_probing(self): ## > performs the probing operations in parallel
        ## define random seed
        seed = np.random.randint(10**8,high=None,size=self.nrun)
        ## get output shape
        result = self.probing(0,field(self.domain,val=None,target=self.target))
        if(np.isscalar(result))or(np.size(result)==1):
            shape = np.ones(1,dtype=np.int,order='C')
        else:
            shape = np.array(np.array(result).shape)
        ## define shared objects
        if(np.iscomplexobj(result)):
            _sum = (ma('d',np.zeros(np.prod(shape,axis=0,dtype=np.int,out=None),dtype=np.float64,order='C'),lock=True),ma('d',np.zeros(np.prod(shape,axis=0,dtype=np.int,out=None),dtype=np.float64,order='C'),lock=True)) ## tuple(real,imag)
        else:
            _sum = ma('d',np.zeros(np.prod(shape,axis=0,dtype=np.int,out=None),dtype=np.float64,order='C'),lock=True)
        _num = mv('I',0,lock=True)
        if(self.var):
            _var = ma('d',np.zeros(np.prod(shape,axis=0,dtype=np.int,out=None),dtype=np.float64,order='C'),lock=True)
        else:
            _var = None
        ## build pool
        if(about.infos.status):
            so.write(about.infos.cstring("INFO: multiprocessing "+(' ')*10))
            so.flush()
        pool = mp(processes=self.ncpu,initializer=_share._init_share,initargs=(_sum,_num,_var),maxtasksperchild=self.nper)
        try:
            ## pooling
            pool.map_async(self._serial_probing,zip(seed,np.arange(self.nrun,dtype=np.int)),chunksize=None,callback=None).get(timeout=None)
            ## close and join pool
            about.infos.cflush(" done.")
            pool.close()
            pool.join()
        except BaseException as exception:
            ## terminate and join pool
            about._errors.cprint("\nERROR: terminating pool.")
            pool.terminate()
            pool.join()
            ## re-raise exception
            raise exception ## traceback by looping
        ## evaluate
        if(np.iscomplexobj(result)):
            _sum = (np.array(_sum[0][:])+np.array(_sum[1][:])*1j).reshape(shape) ## comlpex array
        else:
            _sum = np.array(_sum[:]).reshape(shape)
        if(self.var):
            _var = np.array(_var[:]).reshape(shape)
        return self.evaluate(_sum,_num.value,_var)

    def _nonparallel_probing(self): ## > performs the probing operations one after another
        ## looping
        if(about.infos.status):
            so.write(about.infos.cstring("INFO: looping "+(' ')*10))
            so.flush()
        _sum = 0
        _num = 0
        _var = 0
        for ii in xrange(self.nrun):
            result = self._single_probing((np.random.randint(10**8,high=None,size=None),ii)) ## tuple(seed,idnum)
            if(result is not None):
                _sum += result ## result: {scalar, np.array}
                _num += 1
                if(self.var):
                    _var += np.real(np.conjugate(result)*result)
                self._progress(_num)
        about.infos.cflush(" done.")
        ## evaluate
        return self.evaluate(_sum,_num,_var)

    def __call__(self,loop=False,**kwargs):
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
        if(not about.multiprocessing.status)or(loop):
            return self._nonparallel_probing()
        else:
            return self._parallel_probing()

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty_core.probing>"

##=============================================================================



##-----------------------------------------------------------------------------

class trace_probing(probing):
    """
        ..      __
        ..    /  /_
        ..   /   _/  _____   ____ __   _______   _______
        ..  /  /   /   __/ /   _   / /   ____/ /   __  /
        .. /  /_  /  /    /  /_/  / /  /____  /  /____/
        .. \___/ /__/     \______|  \______/  \______/  probing class

        NIFTY subclass for trace probing (using multiprocessing)

        When called, a trace_probing class instance samples the trace of an
        operator or a function using random fields, whose components are random
        variables with mean 0 and variance 1. When an instance is called it
        returns the mean value of the scalar product of probe and f(probe),
        where probe is a random        field with mean 0 and variance 1.
        The mean is calculated as 1/N Sum[ probe_i.dot(f(probe_i)) ].

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
        probing : The base probing class
        diagonal_probing : A probing class to get the diagonal of an operator
        operator.tr : the trace function uses trace probing in the case of non
            diagonal operators


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
    def __init__(self,op,function=None,domain=None,target=None,random="pm1",ncpu=2,nrun=8,nper=1,var=False,**quargs):
        """
        initializes a trace probing instance

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
        from nifty_operators import operator
        if(not isinstance(op,operator)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        elif(op.nrow()!=op.ncol()):
            raise ValueError(about._errors.cstring("ERROR: trace ill-defined for "+str(op.nrow())+" x "+str(op.ncol())+" matrices."))

        ## check whether callable
        if(function is None)or(not hasattr(function,"__call__")):
            function = op.times
        elif(op==function):
            function = op.times
        ## check whether correctly bound
        if(op!=function.im_self):
            raise NameError(about._errors.cstring("ERROR: invalid input."))
        self.function = function

        ## check given domain
        if(domain is None)or(not isinstance(domain,space)):
            if(self.function in [op.inverse_times,op.adjoint_times]):
                domain = op.target
            else:
                domain = op.domain
        else:
            if(not op.domain.check_codomain(domain)): ## restrictive
                raise ValueError(about._errors.cstring("ERROR: incompatible domains."))
            if(not op.target.check_codomain(domain)): ## restrictive
                raise ValueError(about._errors.cstring("ERROR: incompatible domains."))
            if(target is None)or(not isinstance(target,space)):
                if(function in [op.inverse_times,op.adjoint_times]):
                    target = op.target
                else:
                    target = op.domain
        self.domain = domain

        ## check codomain
        if(target is None):
            target = self.domain.get_codomain()
        else:
            self.domain.check_codomain(target) ## a bit pointless
        self.target = target

        ## check degrees of freedom
        if(op.domain.get_dof()>self.domain.get_dof()):
            about.infos.cprint("INFO: variant numbers of degrees of freedom ( "+str(op.domain.get_dof())+" / "+str(self.domain.get_dof())+" ).")

        if(random not in ["pm1","gau"]):
            raise ValueError(about._errors.cstring("ERROR: unsupported random key '"+str(random)+"'."))
        self.random = random

        self.ncpu = int(max(1,ncpu))
        self.nrun = int(max(self.ncpu,nrun))
        if(nper is None):
            self.nper = None
        else:
            self.nper = int(max(1,min(self.nrun//self.ncpu,nper)))

        self.var = bool(var)

        self.quargs = quargs

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def probing(self,idnum,probe):
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
            result : float
                    the result of `probe.dot(function(probe))`
        """
        f = self.function(probe,**self.quargs) ## field
        if(f is None):
            return None
        else:
            if(f.domain!=self.domain):
                f.transform(target=self.domain,overwrite=True)
            return self.domain.calc_dot(probe.val,f.val) ## discrete inner product

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def evaluate(self,summa,num,var):
        """
            Evaluates the probing results.

            Parameters
            ----------
            summa : scalar
                Sum of all probing results.
            num : int
                Number of successful probings (not returning ``None``).
            var : scalar
                Sum of all squared probing results

            Returns
            -------
            final : scalar
                The final probing result; 1/N Sum[ probing(probe_i) ]
            var : scalar
                The variance of the final probing result;
                1/(N(N-1)) Sum[( probing(probe_i) - final )^2];
                if the variance is returned, the return will be a tuple of
                (`final`,`var`).

        """
        if(num<self.nrun):
            about.infos.cflush(" ( %u probe(s) failed, effectiveness == %.1f%% )\n"%(self.nrun-num,100*num/self.nrun))
            if(num==0):
                about.warnings.cprint("WARNING: probing failed.")
                return None
        else:
            about.infos.cflush("\n")

        if(issubclass(self.domain.datatype,np.complexfloating)):
            summa = np.real(summa)

        final = summa/num
        if(self.var):
            if(num==1):
                about.warnings.cprint("WARNING: infinite variance.")
                return final,None
            else:
                var = var/(num*(num-1))-np.real(np.conjugate(final)*final)/(num-1)
                return final,var
        else:
            return final

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _serial_probing(self,zipped): ## > performs the probing operation serially
        try:
            result = self._single_probing(zipped)
        except Exception as exception:
            raise exception
        except BaseException: ## capture system-exiting exception including KeyboardInterrupt
            raise Exception(about._errors.cstring("ERROR: unknown."))
        else:
            if(result is not None):
                if(isinstance(_share.sum,tuple)):
                    _share.sum[0].acquire(block=True,timeout=None)
                    _share.sum[0].value += np.real(result)
                    _share.sum[0].release()
                    _share.sum[1].acquire(block=True,timeout=None)
                    _share.sum[1].value += np.imag(result)
                    _share.sum[1].release()
                else:
                    _share.sum.acquire(block=True,timeout=None)
                    _share.sum.value += result
                    _share.sum.release()
                _share.num.acquire(block=True,timeout=None)
                _share.num.value += 1
                _share.num.release()
                if(self.var):
                    _share.var.acquire(block=True,timeout=None)
                    _share.var.value += np.real(np.conjugate(result)*result)
                    _share.var.release()
                self._progress(_share.num.value)

    def _parallel_probing(self): ## > performs the probing operations in parallel
        ## define random seed
        seed = np.random.randint(10**8,high=None,size=self.nrun)
        ## define shared objects
        if(issubclass(self.domain.datatype,np.complexfloating)):
            _sum = (mv('d',0,lock=True),mv('d',0,lock=True)) ## tuple(real,imag)
        else:
            _sum = mv('d',0,lock=True)
        _num = mv('I',0,lock=True)
        if(self.var):
            _var = mv('d',0,lock=True)
        else:
            _var = None
        ## build pool
        if(about.infos.status):
            so.write(about.infos.cstring("INFO: multiprocessing "+(' ')*10))
            so.flush()
        pool = mp(processes=self.ncpu,initializer=_share._init_share,initargs=(_sum,_num,_var),maxtasksperchild=self.nper)
        try:
            ## pooling
            pool.map_async(self._serial_probing,zip(seed,np.arange(self.nrun,dtype=np.int)),chunksize=None,callback=None).get(timeout=None)
            ## close and join pool
            about.infos.cflush(" done.")
            pool.close()
            pool.join()
        except BaseException as exception:
            ## terminate and join pool
            about._errors.cprint("\nERROR: terminating pool.")
            pool.terminate()
            pool.join()
            ## re-raise exception
            raise exception ## traceback by looping
        ## evaluate
        if(issubclass(self.domain.datatype,np.complexfloating)):
            _sum = np.complex(_sum[0].value,_sum[1].value)
        else:
            _sum = _sum.value
        if(self.var):
            _var = _var.value
        return self.evaluate(_sum,_num.value,_var)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty_core.trace_probing>"

##-----------------------------------------------------------------------------



##-----------------------------------------------------------------------------

class diagonal_probing(probing):
    """
        ..           __   __                                                     __
        ..         /  / /__/                                                   /  /
        ..    ____/  /  __   ____ __   ____ __   ______    __ ___    ____ __  /  /
        ..  /   _   / /  / /   _   / /   _   / /   _   | /   _   | /   _   / /  /
        .. /  /_/  / /  / /  /_/  / /  /_/  / /  /_/  / /  / /  / /  /_/  / /  /_
        .. \______| /__/  \______|  \___   /  \______/ /__/ /__/  \______|  \___/  probing class
        ..                         /______/

        NIFTY subclass for diagonal probing (using multiprocessing)

        When called, a diagonal_probing class instance samples the diagonal of
        an operator or a function using random fields, whose components are
        random variables with mean 0 and variance 1. When an instance is called
        it returns the mean value of probe*f(probe), where probe is a random
        field with mean 0 and variance 1.
        The mean is calculated as 1/N Sum[ probe_i*f(probe_i) ]
        ('*' denoting component wise multiplication)

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
            the number of cpus to be used for parallel probing. (default: 2)
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
        save : bool, *optional*
            If `save` is True, then the probing results will be written to the
            hard disk instead of being saved in the RAM. This is recommended
            for high dimensional fields whose probes would otherwise fill up
            the memory. (default: False)
        path : string, *optional*
            the path, where the probing results are saved, if `save` is True.
            (default: "tmp")
        prefix : string, *optional*
            a prefix for the saved probing results. The saved results will be
            named using that prefix and an 8-digit number
            (e.g. "<prefix>00000001.npy"). (default: "")


        See Also
        --------
        trace_probing : A probing class to get the trace of an operator
        probing : The base probing class
        operator.diag : The diag function uses diagonal probing in the case of
            non diagonal operators
        operator.hat : The hat function uses diagonal probing in the case of
            non diagonal operators


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
        save : {string, None}
            the path and prefix for saved probe files. None in the case where
            the probing results are stored in the RAM.
        quargs : dict
            Keyword arguments passed to `function` in each call.

    """
    def __init__(self,op,function=None,domain=None,target=None,random="pm1",ncpu=2,nrun=8,nper=1,var=False,save=False,path="tmp",prefix="",**quargs):
        """
        initializes a diagonal probing instance

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
            the number of cpus to be used for parallel probing. (default: 2)
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
            It is recommended to leave `nper` as the default value. (default: 8)
        var : bool, *optional*
            If `var` is True, then the variance of the sampled function will
            also be returned. The result is then a tuple with the mean in the
            zeroth entry and the variance in the first entry. (default: False)
        save : bool, *optional*
            If `save` is True, then the probing results will be written to the
            hard disk instead of being saved in the RAM. This is recommended
            for high dimensional fields whose probes would otherwise fill up
            the memory. (default: False)
        path : string, *optional*
            the path, where the probing results are saved, if `save` is True.
            (default: "tmp")
        prefix : string, *optional*
            a prefix for the saved probing results. The saved results will be
            named using that prefix and an 8-digit number
            (e.g. "<prefix>00000001.npy"). (default: "")

        """
        from nifty_operators import operator
        if(not isinstance(op,operator)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        elif(op.nrow()!=op.ncol()):
            raise ValueError(about._errors.cstring("ERROR: diagonal ill-defined for "+str(op.nrow())+" x "+str(op.ncol())+" matrices."))

        ## check whether callable
        if(function is None)or(not hasattr(function,"__call__")):
            function = op.times
        elif(op==function):
            function = op.times
        ## check whether correctly bound
        if(op!=function.im_self):
            raise NameError(about._errors.cstring("ERROR: invalid input."))
        self.function = function

        ## check given domain
        if(domain is None)or(not isinstance(domain,space)):
            if(self.function in [op.inverse_times,op.adjoint_times]):
                domain = op.target
            else:
                domain = op.domain
        else:
            if(not op.domain.check_codomain(domain)): ## restrictive
                raise ValueError(about._errors.cstring("ERROR: incompatible domains."))
            if(not op.target.check_codomain(domain)): ## restrictive
                raise ValueError(about._errors.cstring("ERROR: incompatible domains."))
            if(target is None)or(not isinstance(target,space)):
                if(function in [op.inverse_times,op.adjoint_times]):
                    target = op.target
                else:
                    target = op.domain
        self.domain = domain

        ## check codomain
        if(target is None):
            target = self.domain.get_codomain()
        else:
            self.domain.check_codomain(target) ## a bit pointless
        self.target = target

        ## check degrees of freedom
        if(self.domain.get_dof()>op.domain.get_dof()):
            about.infos.cprint("INFO: variant numbers of degrees of freedom ( "+str(self.domain.get_dof())+" / "+str(op.domain.get_dof())+" ).")

        if(random not in ["pm1","gau"]):
            raise ValueError(about._errors.cstring("ERROR: unsupported random key '"+str(random)+"'."))
        self.random = random

        self.ncpu = int(max(1,ncpu))
        self.nrun = int(max(self.ncpu,nrun))
        if(nper is None):
            self.nper = None
        else:
            self.nper = int(max(1,min(self.nrun//self.ncpu,nper)))

        self.var = bool(var)

        if(save):
            path = os.path.expanduser(str(path))
            if(not os.path.exists(path)):
                os.makedirs(path)
            self.save = os.path.join(path,str(prefix)) ## (back)slash inserted if needed
        else:
            self.save = None

        self.quargs = quargs

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def configure(self,**kwargs):
        """
            changes the attributes of the instance

            Parameters
            ----------
            random : string, *optional*
                the random number generator used to create the probes
                (default: "pm1")
            ncpu : int, *optional*
                the number of cpus to be used for parallel probing
                (default: 2)
            nrun : int, *optional*
                the number of probes to be evaluated. If `nrun<ncpu`, it will
                be set to `ncpu`. (default: 8)
            nper : int, *optional*
                number of probes, that will be evaluated by one worker
                (default: 1)
            var : bool, *optional*
                whether the variance will be additionally returned
                (default: False)
            save : bool, *optional*
                whether the individual probing results will be saved to the HDD
                (default: False)
            path : string, *optional*
                the path, where the probing results are saved (default: "tmp")
            prefix : string, *optional*
                a prefix for the saved probing results (default: "")

        """
        if("random" in kwargs):
            if(kwargs.get("random") not in ["pm1","gau"]):
                raise ValueError(about._errors.cstring("ERROR: unsupported random key '"+str(kwargs.get("random"))+"'."))
            self.random = kwargs.get("random")

        if("ncpu" in kwargs):
            self.ncpu = int(max(1,kwargs.get("ncpu")))
        if("nrun" in kwargs):
            self.nrun = int(max(self.ncpu,kwargs.get("nrun")))
        if("nper" in kwargs):
            if(kwargs.get("nper") is None):
                self.nper = None
            else:
                self.nper = int(max(1,min(self.nrun//self.ncpu,kwargs.get("nper"))))

        if("var" in kwargs):
            self.var = bool(kwargs.get("var"))

        if("save" in kwargs):
            if(kwargs.get("save")):
                if("path" in kwargs):
                    path = kwargs.get("path")
                else:
                    if(self.save is not None):
                        about.warnings.cprint("WARNING: save path set to default.")
                    path = "tmp"
                if("prefix" in kwargs):
                    prefix = kwargs.get("prefix")
                else:
                    if(self.save is not None):
                        about.warnings.cprint("WARNING: save prefix set to default.")
                    prefix = ""
                path = os.path.expanduser(str(path))
                if(not os.path.exists(path)):
                    os.makedirs(path)
                self.save = os.path.join(path,str(prefix)) ## (back)slash inserted if needed
            else:
                self.save = None

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def probing(self,idnum,probe):

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
            result : ndarray
                    the result of `probe*(function(probe))`
        """
        f = self.function(probe,**self.quargs) ## field
        if(f is None):
            return None
        else:
            if(f.domain!=self.domain):
                f.transform(target=self.domain,overwrite=True)
            result = np.conjugate(probe.val)*f.val
            if(self.save is not None):
                np.save(self.save+"%08u"%idnum,result)
            return result

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _serial_probing(self,zipped): ## > performs the probing operation serially
        try:
            result = self._single_probing(zipped)
        except Exception as exception:
            raise exception
        except BaseException: ## capture system-exiting exception including KeyboardInterrupt
            raise Exception(about._errors.cstring("ERROR: unknown."))
        else:
            if(result is not None):
                result = np.array(result).flatten(order='C')
                if(isinstance(_share.sum,tuple)):
                    _share.sum[0].acquire(block=True,timeout=None)
                    _share.sum[0][:] += np.real(result)
                    _share.sum[0].release()
                    _share.sum[1].acquire(block=True,timeout=None)
                    _share.sum[1][:] += np.imag(result)
                    _share.sum[1].release()
                else:
                    _share.sum.acquire(block=True,timeout=None)
                    _share.sum[:] += result
                    _share.sum.release()
                _share.num.acquire(block=True,timeout=None)
                _share.num.value += 1
                _share.num.release()
                if(self.var):
                    _share.var.acquire(block=True,timeout=None)
                    _share.var[:] += np.real(np.conjugate(result)*result)
                    _share.var.release()
                self._progress(_share.num.value)

    def _parallel_probing(self): ## > performs the probing operations in parallel
        ## define random seed
        seed = np.random.randint(10**8,high=None,size=self.nrun)
        ## define shared objects
        if(issubclass(self.domain.datatype,np.complexfloating)):
            _sum = (ma('d',np.zeros(self.domain.get_dim(split=False),dtype=np.float64,order='C'),lock=True),ma('d',np.zeros(self.domain.get_dim(split=False),dtype=np.float64,order='C'),lock=True)) ## tuple(real,imag)
        else:
            _sum = ma('d',np.zeros(self.domain.get_dim(split=False),dtype=np.float64,order='C'),lock=True)
        _num = mv('I',0,lock=True)
        if(self.var):
            _var = ma('d',np.zeros(self.domain.get_dim(split=False),dtype=np.float64,order='C'),lock=True)
        else:
            _var = None
        ## build pool
        if(about.infos.status):
            so.write(about.infos.cstring("INFO: multiprocessing "+(' ')*10))
            so.flush()
        pool = mp(processes=self.ncpu,initializer=_share._init_share,initargs=(_sum,_num,_var),maxtasksperchild=self.nper)
        try:
            ## pooling
            pool.map_async(self._serial_probing,zip(seed,np.arange(self.nrun,dtype=np.int)),chunksize=None,callback=None).get(timeout=None)
            ## close and join pool
            about.infos.cflush(" done.")
            pool.close()
            pool.join()
        except BaseException as exception:
            ## terminate and join pool
            about._errors.cprint("\nERROR: terminating pool.")
            pool.terminate()
            pool.join()
            ## re-raise exception
            raise exception ## traceback by looping
        ## evaluate
        if(issubclass(self.domain.datatype,np.complexfloating)):
            _sum = (np.array(_sum[0][:])+np.array(_sum[1][:])*1j).reshape(self.domain.get_dim(split=True)) ## comlpex array
        else:
            _sum = np.array(_sum[:]).reshape(self.domain.get_dim(split=True))
        if(self.var):
            _var = np.array(_var[:]).reshape(self.domain.get_dim(split=True))
        return self.evaluate(_sum,_num.value,_var)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty_core.diagonal_probing>"

##-----------------------------------------------------------------------------

## IDEA: diagonal_inference
