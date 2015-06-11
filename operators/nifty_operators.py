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
import numpy as np
from nifty.nifty_about import about
from nifty.nifty_core import space, \
                         point_space, \
                         nested_space, \
                         field
from nifty.nifty_tools import conjugate_gradient
from nifty_probing import trace_probing, \
                            diagonal_probing

##=============================================================================

class operator(object):
    """
        ..                                                      __
        ..                                                    /  /_
        ..    ______    ______    _______   _____   ____ __  /   _/  ______    _____
        ..  /   _   | /   _   | /   __  / /   __/ /   _   / /  /   /   _   | /   __/
        .. /  /_/  / /  /_/  / /  /____/ /  /    /  /_/  / /  /_  /  /_/  / /  /
        .. \______/ /   ____/  \______/ /__/     \______|  \___/  \______/ /__/     class
        ..         /__/

        NIFTY base class for (linear) operators

        The base NIFTY operator class is an abstract class from which other
        specific operator subclasses, including those preimplemented in NIFTY
        (e.g. the diagonal operator class) must be derived.

        Parameters
        ----------
        domain : space
            The space wherein valid arguments live.
        sym : bool, *optional*
            Indicates whether the operator is self-adjoint or not
            (default: False)
        uni : bool, *optional*
            Indicates whether the operator is unitary or not
            (default: False)
        imp : bool, *optional*
            Indicates whether the incorporation of volume weights in
            multiplications is already implemented in the `multiply`
            instance methods or not (default: False)
        target : space, *optional*
            The space wherein the operator output lives (default: domain)
        para : {single object, list of objects}, *optional*
            This is a freeform list of parameters that derivatives of the
            operator class can use. Not used in the base operators.
            (default: None)

        See Also
        --------
        diagonal_operator :  An operator class for handling purely diagonal
            operators.
        power_operator : Similar to diagonal_operator but with handy features
            for dealing with diagonal operators whose diagonal
            consists of a power spectrum.
        vecvec_operator : Operators constructed from the outer product of two
            fields
        response_operator : Implements a modeled instrument response which
            translates a signal into data space.
        projection_operator : An operator that projects out one or more
            components in a basis, e.g. a spectral band
            of Fourier components.

        Attributes
        ----------
        domain : space
            The space wherein valid arguments live.
        sym : bool
            Indicates whether the operator is self-adjoint or not
        uni : bool
            Indicates whether the operator is unitary or not
        imp : bool
            Indicates whether the incorporation of volume weights in
            multiplications is already implemented in the `multiply`
            instance methods or not
        target : space
            The space wherein the operator output lives
        para : {single object, list of objects}
            This is a freeform list of parameters that derivatives of the
            operator class can use. Not used in the base operators.
    """
    def __init__(self,domain,sym=False,uni=False,imp=False,target=None,para=None):
        """
            Sets the attributes for an operator class instance.

            Parameters
            ----------
            domain : space
                The space wherein valid arguments live.
            sym : bool, *optional*
                Indicates whether the operator is self-adjoint or not
                (default: False)
            uni : bool, *optional*
                Indicates whether the operator is unitary or not
                (default: False)
            imp : bool, *optional*
                Indicates whether the incorporation of volume weights in
                multiplications is already implemented in the `multiply`
                instance methods or not (default: False)
            target : space, *optional*
                The space wherein the operator output lives (default: domain)
            para : {object, list of objects}, *optional*
                This is a freeform list of parameters that derivatives of the
                operator class can use. Not used in the base operators.
                (default: None)

            Returns
            -------
            None
        """
        if(not isinstance(domain,space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        self.domain = domain
        self.sym = bool(sym)
        self.uni = bool(uni)

        if(target is None)or(self.sym)or(self.uni):
            target = self.domain
        elif(not isinstance(target,space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        self.target = target

        if(self.domain.discrete)and(self.target.discrete):
            self.imp = True
        else:
            self.imp = bool(imp)

        if(para is not None):
            self.para = para

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def nrow(self):
        """
            Computes the number of rows.

            Returns
            -------
            nrow : int
                number of rows (equal to the dimension of the codomain)
        """
        return self.target.dim(split=False)

    def ncol(self):
        """
            Computes the number of columns

            Returns
            -------
            ncol : int
                number of columns (equal to the dimension of the domain)
        """
        return self.domain.dim(split=False)

    def dim(self,axis=None):
        """
            Computes the dimension of the space

            Parameters
            ----------
            axis : int, *optional*
                Axis along which the dimension is to be calculated.
                (default: None)

            Returns
            -------
            dim : {int, ndarray}
                The dimension(s) of the operator.

        """
        if(axis is None):
            return np.array([self.nrow(),self.ncol()])
        elif(axis==0):
            return self.nrow()
        elif(axis==1):
            return self.ncol()
        else:
            raise ValueError(about._errors.cstring("ERROR: invalid input axis."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def set_para(self,newpara):
        """
            Sets the parameters and creates the `para` property if it does
            not exist

            Parameters
            ----------
            newpara : {object, list of objects}
                A single parameter or a list of parameters.

            Returns
            -------
            None

        """
        self.para = newpara

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _multiply(self,x,**kwargs): ## > applies the operator to a given field
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'multiply'."))

    def _adjoint_multiply(self,x,**kwargs): ## > applies the adjoint operator to a given field
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'adjoint_multiply'."))

    def _inverse_multiply(self,x,**kwargs): ## > applies the inverse operator to a given field
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'inverse_multiply'."))

    def _adjoint_inverse_multiply(self,x,**kwargs): ## > applies the inverse adjoint operator to a given field
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'adjoint_inverse_multiply'."))

    def _inverse_adjoint_multiply(self,x,**kwargs): ## > applies the adjoint inverse operator to a given field
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'inverse_adjoint_multiply'."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _briefing(self,x,domain,inverse): ## > prepares x for `multiply`
        ## inspect x
        if(not isinstance(x,field)):
            x_ = field(domain,val=x,target=None)
        else:
            ## check x.domain
            if(domain==x.domain):
                x_ = x
            ## transform
            else:
                x_ = x.transform(target=domain,overwrite=False)
        ## weight if ...
        if(not self.imp)and(not domain.discrete)and(not inverse):
            x_ = x_.weight(power=1,overwrite=False)
        return x_

    def _debriefing(self,x,x_,target,inverse): ## > evaluates x and x_ after `multiply`
        if(x_ is None):
            return None
        else:
            ## inspect x_
            if(not isinstance(x_,field)):
                x_ = field(target,val=x_,target=None)
            elif(x_.domain!=target):
                raise ValueError(about._errors.cstring("ERROR: invalid output domain."))
            ## weight if ...
            if(not self.imp)and(not target.discrete)and(inverse):
                x_ = x_.weight(power=-1,overwrite=False)
            ## inspect x
            if(isinstance(x,field)):
                ## repair ...
                if(self.domain==self.target!=x.domain):
                    x_ = x_.transform(target=x.domain,overwrite=False) ## ... domain
                if(x_.domain==x.domain)and(x_.target!=x.target):
                    x_.set_target(newtarget=x.target) ## ... codomain
            return x_

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def times(self,x,**kwargs):
        """
            Applies the operator to a given object

            Parameters
            ----------
            x : {scalar, ndarray, field}
                Scalars are interpreted as constant arrays, and an array will
                be interpreted as a field on the domain of the operator.

            Returns
            -------
            Ox : field
                Mapped field on the target domain of the operator.
        """
        ## prepare
        x_ = self._briefing(x,self.domain,False)
        ## apply operator
        x_ = self._multiply(x_,**kwargs)
        ## evaluate
        return self._debriefing(x,x_,self.target,False)

    def __call__(self,x,**kwargs):
        return self.times(x,**kwargs)

    def adjoint_times(self,x,**kwargs):
        """
            Applies the adjoint operator to a given object.

            Parameters
            ----------
            x : {scalar, ndarray, field}
                Scalars are interpreted as constant arrays, and an array will
                be interpreted as a field on the target space of the operator.

            Returns
            -------
            OAx : field
                Mapped field on the domain of the operator.

        """
        ## check whether self-adjoint
        if(self.sym):
            return self.times(x,**kwargs)
        ## check whether unitary
        if(self.uni):
            return self.inverse_times(x,**kwargs)

        ## prepare
        x_ = self._briefing(x,self.target,False)
        ## apply operator
        x_ = self._adjoint_multiply(x_,**kwargs)
        ## evaluate
        return self._debriefing(x,x_,self.domain,False)

    def inverse_times(self,x,**kwargs):
        """
            Applies the inverse operator to a given object.

            Parameters
            ----------
            x : {scalar, ndarray, field}
                Scalars are interpreted as constant arrays, and an array will
                be interpreted as a field on the domain space of the operator.

            Returns
            -------
            OIx : field
                Mapped field on the target space of the operator.
        """
        ## check whether self-inverse
        if(self.sym)and(self.uni):
            return self.times(x,**kwargs)

        ## prepare
        x_ = self._briefing(x,self.target,True)
        ## apply operator
        x_ = self._inverse_multiply(x_,**kwargs)
        ## evaluate
        return self._debriefing(x,x_,self.domain,True)

    def adjoint_inverse_times(self,x,**kwargs):
        """
            Applies the inverse adjoint operator to a given object.

            Parameters
            ----------
            x : {scalar, ndarray, field}
                Scalars are interpreted as constant arrays, and an array will
                be interpreted as a field on the target space of the operator.

            Returns
            -------
            OAIx : field
                Mapped field on the domain of the operator.

        """
        ## check whether self-adjoint
        if(self.sym):
            return self.inverse_times(x,**kwargs)
        ## check whether unitary
        if(self.uni):
            return self.times(x,**kwargs)

        ## prepare
        x_ = self._briefing(x,self.domain,True)
        ## apply operator
        x_ = self._adjoint_inverse_multiply(x_,**kwargs)
        ## evaluate
        return self._debriefing(x,x_,self.target,True)

    def inverse_adjoint_times(self,x,**kwargs):
        """
            Applies the adjoint inverse operator to a given object.

            Parameters
            ----------
            x : {scalar, ndarray, field}
                Scalars are interpreted as constant arrays, and an array will
                be interpreted as a field on the target space of the operator.

            Returns
            -------
            OIAx : field
                Mapped field on the domain of the operator.

        """
        ## check whether self-adjoint
        if(self.sym):
            return self.inverse_times(x,**kwargs)
        ## check whether unitary
        if(self.uni):
            return self.times(x,**kwargs)

        ## prepare
        x_ = self._briefing(x,self.domain,True)
        ## apply operator
        x_ = self._inverse_adjoint_multiply(x_,**kwargs)
        ## evaluate
        return self._debriefing(x,x_,self.target,True)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def tr(self,domain=None,target=None,random="pm1",ncpu=2,nrun=8,nper=1,var=False,loop=False,**kwargs):
        """
            Computes the trace of the operator

            Parameters
            ----------
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: 1)
            var : bool, *optional*
                Indicates whether to additionally return the probing variance
                or not (default: False).
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            tr : float
                Trace of the operator
            delta : float, *optional*
                Probing variance of the trace. Returned if `var` is True in
                of probing case.

            See Also
            --------
            probing : The class used to perform probing operations
        """
        if(domain is None):
            domain = self.domain
        return trace_probing(self,function=self.times,domain=domain,target=target,random=random,ncpu=(ncpu,1)[bool(loop)],nrun=nrun,nper=nper,var=var,**kwargs)(loop=loop)

    def inverse_tr(self,domain=None,target=None,random="pm1",ncpu=2,nrun=8,nper=1,var=False,loop=False,**kwargs):
        """
            Computes the trace of the inverse operator

            Parameters
            ----------
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: 1)
            var : bool, *optional*
                Indicates whether to additionally return the probing variance
                or not (default: False).
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            tr : float
                Trace of the inverse operator
            delta : float, *optional*
                Probing variance of the trace. Returned if `var` is True in
                of probing case.

            See Also
            --------
            probing : The class used to perform probing operations
        """
        if(domain is None):
            domain = self.target
        return trace_probing(self,function=self.inverse_times,domain=domain,target=target,random=random,ncpu=(ncpu,1)[bool(loop)],nrun=nrun,nper=nper,var=var,**kwargs)(loop=loop)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def diag(self,bare=False,domain=None,target=None,random="pm1",ncpu=2,nrun=8,nper=1,var=False,save=False,path="tmp",prefix="",loop=False,**kwargs):
        """
            Computes the diagonal of the operator via probing.

            Parameters
            ----------
            bare : bool, *optional*
                Indicatese whether the diagonal entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: False)
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: 1)
            var : bool, *optional*
                Indicates whether to additionally return the probing variance
                or not (default: False).
            save : bool, *optional*
                whether all individual probing results are saved or not
                (default: False)
            path : string, *optional*
                path wherein the results are saved (default: "tmp")
            prefix : string, *optional*
                prefix for all saved files (default: "")
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            diag : ndarray
                The matrix diagonal
            delta : float, *optional*
                Probing variance of the trace. Returned if `var` is True in
                of probing case.

            See Also
            --------
            probing : The class used to perform probing operations

            Notes
            -----
            The ambiguity of `bare` or non-bare diagonal entries is based
            on the choice of a matrix representation of the operator in
            question. The naive choice of absorbing the volume weights
            into the matrix leads to a matrix-vector calculus with the
            non-bare entries which seems intuitive, though. The choice of
            keeping matrix entries and volume weights separate deals with the
            bare entries that allow for correct interpretation of the matrix
            entries; e.g., as variance in case of an covariance operator.

        """
        if(domain is None):
            domain = self.domain
        diag = diagonal_probing(self,function=self.times,domain=domain,target=target,random=random,ncpu=(ncpu,1)[bool(loop)],nrun=nrun,nper=nper,var=var,save=save,path=path,prefix=prefix,**kwargs)(loop=loop)
        if(diag is None):
#            about.warnings.cprint("WARNING: forwarding 'NoneType'.")
            return None
        ## weight if ...
        elif(not domain.discrete)and(bare):
            if(isinstance(diag,tuple)): ## diag == (diag,variance)
                return domain.calc_weight(diag[0],power=-1),domain.calc_weight(diag[1],power=-1)
            else:
                return domain.calc_weight(diag,power=-1)
        else:
            return diag

    def inverse_diag(self,bare=False,domain=None,target=None,random="pm1",ncpu=2,nrun=8,nper=1,var=False,save=False,path="tmp",prefix="",loop=False,**kwargs):
        """
            Computes the diagonal of the inverse operator via probing.

            Parameters
            ----------
            bare : bool, *optional*
                Indicatese whether the diagonal entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: False)
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: 1)
            var : bool, *optional*
                Indicates whether to additionally return the probing variance
                or not (default: False).
            save : bool, *optional*
                whether all individual probing results are saved or not
                (default: False)
            path : string, *optional*
                path wherein the results are saved (default: "tmp")
            prefix : string, *optional*
                prefix for all saved files (default: "")
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            diag : ndarray
                The diagonal of the inverse matrix
            delta : float, *optional*
                Probing variance of the trace. Returned if `var` is True in
                of probing case.

            See Also
            --------
            probing : The class used to perform probing operations

            Notes
            -----
            The ambiguity of `bare` or non-bare diagonal entries is based
            on the choice of a matrix representation of the operator in
            question. The naive choice of absorbing the volume weights
            into the matrix leads to a matrix-vector calculus with the
            non-bare entries which seems intuitive, though. The choice of
            keeping matrix entries and volume weights separate deals with the
            bare entries that allow for correct interpretation of the matrix
            entries; e.g., as variance in case of an covariance operator.

        """
        if(domain is None):
            domain = self.target
        diag = diagonal_probing(self,function=self.inverse_times,domain=domain,target=target,random=random,ncpu=(ncpu,1)[bool(loop)],nrun=nrun,nper=nper,var=var,save=save,path=path,prefix=prefix,**kwargs)(loop=loop)
        if(diag is None):
#            about.warnings.cprint("WARNING: forwarding 'NoneType'.")
            return None
        ## weight if ...
        elif(not domain.discrete)and(bare):
            if(isinstance(diag,tuple)): ## diag == (diag,variance)
                return domain.calc_weight(diag[0],power=-1),domain.calc_weight(diag[1],power=-1)
            else:
                return domain.calc_weight(diag,power=-1)
        else:
            return diag

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def det(self):
        """
            Computes the determinant of the operator.

            Returns
            -------
            det : float
                The determinant

        """
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'det'."))

    def inverse_det(self):
        """
            Computes the determinant of the inverse operator.

            Returns
            -------
            det : float
                The determinant

        """
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'inverse_det'."))

    def log_det(self):
        """
            Computes the logarithm of the determinant of the operator (if applicable).

            Returns
            -------
            logdet : float
                The logarithm of the determinant

        """
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'log_det'."))

    def tr_log(self):
        """
            Computes the trace of the logarithm of the operator (if applicable).

            Returns
            -------
            logdet : float
                The trace of the logarithm

        """
        return self.log_det()

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def hat(self,bare=False,domain=None,target=None,**kwargs):
        """
            Translates the operator's diagonal into a field

            Parameters
            ----------
            bare : bool, *optional*
                Indicatese whether the diagonal entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: False)
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: 1)
            save : bool, *optional*
                whether all individual probing results are saved or not
                (default: False)
            path : string, *optional*
                path wherein the results are saved (default: "tmp")
            prefix : string, *optional*
                prefix for all saved files (default: "")
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            x : field
                The matrix diagonal as a field living on the operator
                domain space

            See Also
            --------
            probing : The class used to perform probing operations

            Notes
            -----
            The ambiguity of `bare` or non-bare diagonal entries is based
            on the choice of a matrix representation of the operator in
            question. The naive choice of absorbing the volume weights
            into the matrix leads to a matrix-vector calculus with the
            non-bare entries which seems intuitive, though. The choice of
            keeping matrix entries and volume weights separate deals with the
            bare entries that allow for correct interpretation of the matrix
            entries; e.g., as variance in case of an covariance operator.

        """
        if(domain is None):
            domain = self.domain
        diag = self.diag(bare=bare,domain=domain,target=target,var=False,**kwargs)
        if(diag is None):
            about.warnings.cprint("WARNING: forwarding 'NoneType'.")
            return None
        return field(domain,val=diag,target=target)

    def inverse_hat(self,bare=False,domain=None,target=None,**kwargs):
        """
            Translates the inverse operator's diagonal into a field

            Parameters
            ----------
            bare : bool, *optional*
                Indicatese whether the diagonal entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: False)
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: 1)
            save : bool, *optional*
                whether all individual probing results are saved or not
                (default: False)
            path : string, *optional*
                path wherein the results are saved (default: "tmp")
            prefix : string, *optional*
                prefix for all saved files (default: "")
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            x : field
                The matrix diagonal as a field living on the operator
                domain space

            See Also
            --------
            probing : The class used to perform probing operations

            Notes
            -----
            The ambiguity of `bare` or non-bare diagonal entries is based
            on the choice of a matrix representation of the operator in
            question. The naive choice of absorbing the volume weights
            into the matrix leads to a matrix-vector calculus with the
            non-bare entries which seems intuitive, though. The choice of
            keeping matrix entries and volume weights separate deals with the
            bare entries that allow for correct interpretation of the matrix
            entries; e.g., as variance in case of an covariance operator.

        """
        if(domain is None):
            domain = self.target
        diag = self.inverse_diag(bare=bare,domain=domain,target=target,var=False,**kwargs)
        if(diag is None):
            about.warnings.cprint("WARNING: forwarding 'NoneType'.")
            return None
        return field(domain,val=diag,target=target)

    def hathat(self,domain=None,**kwargs):
        """
            Translates the operator's diagonal into a diagonal operator

            Parameters
            ----------
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: 1)
            save : bool, *optional*
                whether all individual probing results are saved or not
                (default: False)
            path : string, *optional*
                path wherein the results are saved (default: "tmp")
            prefix : string, *optional*
                prefix for all saved files (default: "")
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            D : diagonal_operator
                The matrix diagonal as an operator

            See Also
            --------
            probing : The class used to perform probing operations

            Notes
            -----
            The ambiguity of `bare` or non-bare diagonal entries is based
            on the choice of a matrix representation of the operator in
            question. The naive choice of absorbing the volume weights
            into the matrix leads to a matrix-vector calculus with the
            non-bare entries which seems intuitive, though. The choice of
            keeping matrix entries and volume weights separate deals with the
            bare entries that allow for correct interpretation of the matrix
            entries; e.g., as variance in case of an covariance operator.

        """
        if(domain is None):
            domain = self.domain
        diag = self.diag(bare=False,domain=domain,var=False,**kwargs)
        if(diag is None):
            about.warnings.cprint("WARNING: forwarding 'NoneType'.")
            return None
        return diagonal_operator(domain=domain,diag=diag,bare=False)

    def inverse_hathat(self,domain=None,**kwargs):
        """
            Translates the inverse operator's diagonal into a diagonal
            operator

            Parameters
            ----------
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: 1)
            save : bool, *optional*
                whether all individual probing results are saved or not
                (default: False)
            path : string, *optional*
                path wherein the results are saved (default: "tmp")
            prefix : string, *optional*
                prefix for all saved files (default: "")
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            D : diagonal_operator
                The diagonal of the inverse matrix as an operator

            See Also
            --------
            probing : The class used to perform probing operations

            Notes
            -----
            The ambiguity of `bare` or non-bare diagonal entries is based
            on the choice of a matrix representation of the operator in
            question. The naive choice of absorbing the volume weights
            into the matrix leads to a matrix-vector calculus with the
            non-bare entries which seems intuitive, though. The choice of
            keeping matrix entries and volume weights separate deals with the
            bare entries that allow for correct interpretation of the matrix
            entries; e.g., as variance in case of an covariance operator.

        """
        if(domain is None):
            domain = self.target
        diag = self.inverse_diag(bare=False,domain=domain,var=False,**kwargs)
        if(diag is None):
            about.warnings.cprint("WARNING: forwarding 'NoneType'.")
            return None
        return diagonal_operator(domain=domain,diag=diag,bare=False)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty_core.operator>"

##=============================================================================



##-----------------------------------------------------------------------------

class diagonal_operator(operator):
    """
        ..           __   __                                                     __
        ..         /  / /__/                                                   /  /
        ..    ____/  /  __   ____ __   ____ __   ______    __ ___    ____ __  /  /
        ..  /   _   / /  / /   _   / /   _   / /   _   | /   _   | /   _   / /  /
        .. /  /_/  / /  / /  /_/  / /  /_/  / /  /_/  / /  / /  / /  /_/  / /  /_
        .. \______| /__/  \______|  \___   /  \______/ /__/ /__/  \______|  \___/  operator class
        ..                         /______/

        NIFTY subclass for diagonal operators

        Parameters
        ----------
        domain : space, *optional*
            The space wherein valid arguments live. If no domain is given
            then the diag parameter *must* be a field and the domain
            of that field is assumed. (default: None)
        diag : {scalar, ndarray, field}
            The diagonal entries of the operator. For a scalar, a constant
            diagonal is defined having the value provided. If no domain
            is given, diag must be a field. (default: 1)
        bare : bool, *optional*
            whether the diagonal entries are `bare` or not
            (mandatory for the correct incorporation of volume weights)
            (default: False)

        Notes
        -----
        The ambiguity of `bare` or non-bare diagonal entries is based
        on the choice of a matrix representation of the operator in
        question. The naive choice of absorbing the volume weights
        into the matrix leads to a matrix-vector calculus with the
        non-bare entries which seems intuitive, though. The choice of
        keeping matrix entries and volume weights separate deals with the
        bare entries that allow for correct interpretation of the matrix
        entries; e.g., as variance in case of an covariance operator.

        The inverse applications of the diagonal operator feature a ``pseudo``
        flag indicating if zero divison shall be ignored and return zero
        instead of causing an error.

        Attributes
        ----------
        domain : space
            The space wherein valid arguments live.
        val : ndarray
            A field containing the diagonal entries of the matrix.
        sym : bool
            Indicates whether the operator is self-adjoint or not
        uni : bool
            Indicates whether the operator is unitary or not
        imp : bool
            Indicates whether the incorporation of volume weights in
            multiplications is already implemented in the `multiply`
            instance methods or not
        target : space
            The space wherein the operator output lives
    """
    def __init__(self,domain=None,diag=1,bare=False):
        """
            Sets the standard operator properties and `values`.

            Parameters
            ----------
            domain : space, *optional*
                The space wherein valid arguments live. If no domain is given
                then the diag parameter *must* be a field and the domain
                of that field is assumed. (default: None)
            diag : {scalar, ndarray, field}, *optional*
                The diagonal entries of the operator. For a scalar, a constant
                diagonal is defined having the value provided. If no domain
                is given, diag must be a field. (default: 1)
            bare : bool, *optional*
                whether the diagonal entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: False)

            Returns
            -------
            None

            Notes
            -----
            The ambiguity of `bare` or non-bare diagonal entries is based
            on the choice of a matrix representation of the operator in
            question. The naive choice of absorbing the volume weights
            into the matrix leads to a matrix-vector calculus with the
            non-bare entries which seems intuitive, though. The choice of
            keeping matrix entries and volume weights separate deals with the
            bare entries that allow for correct interpretation of the matrix
            entries; e.g., as variance in case of an covariance operator.

        """
        if(domain is None)and(isinstance(diag,field)):
            domain = diag.domain
        if(not isinstance(domain,space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        self.domain = domain

        diag = self.domain.enforce_values(diag,extend=True)
        ## weight if ...
        if(not self.domain.discrete)and(bare):
            diag = self.domain.calc_weight(diag,power=1)
        ## check complexity
        if(np.all(np.imag(diag)==0)):
            self.val = np.real(diag)
            self.sym = True
        else:
            self.val = diag
#            about.infos.cprint("INFO: non-self-adjoint complex diagonal operator.")
            self.sym = False

        ## check whether identity
        if(np.all(diag==1)):
            self.uni = True
        else:
            self.uni = False

        self.imp = True ## correctly implemented for efficiency
        self.target = self.domain

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def set_diag(self,newdiag,bare=False):
        """
            Sets the diagonal of the diagonal operator

            Parameters
            ----------
            newdiag : {scalar, ndarray, field}
                The new diagonal entries of the operator. For a scalar, a
                constant diagonal is defined having the value provided. If
                no domain is given, diag must be a field.

            bare : bool, *optional*
                whether the diagonal entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: False)

            Returns
            -------
            None
        """
        newdiag = self.domain.enforce_values(newdiag,extend=True)
        ## weight if ...
        if(not self.domain.discrete)and(bare):
            newdiag = self.domain.calc_weight(newdiag,power=1)
        ## check complexity
        if(np.all(np.imag(newdiag)==0)):
            self.val = np.real(newdiag)
            self.sym = True
        else:
            self.val = newdiag
#            about.infos.cprint("INFO: non-self-adjoint complex diagonal operator.")
            self.sym = False

        ## check whether identity
        if(np.all(newdiag==1)):
            self.uni = True
        else:
            self.uni = False

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _multiply(self,x,**kwargs): ## > applies the operator to a given field
        x_ = field(self.target,val=None,target=x.target)
        x_.val = x.val*self.val ## bypasses self.domain.enforce_values
        return x_

    def _adjoint_multiply(self,x,**kwargs): ## > applies the adjoint operator to a given field
        x_ = field(self.domain,val=None,target=x.target)
        x_.val = x.val*np.conjugate(self.val) ## bypasses self.domain.enforce_values
        return x_

    def _inverse_multiply(self,x,pseudo=False,**kwargs): ## > applies the inverse operator to a given field
        if(np.any(self.val==0)):
            if(pseudo):
                x_ = field(self.domain,val=None,target=x.target)
                x_.val = np.ma.filled(x.val/np.ma.masked_where(self.val==0,self.val,copy=False),fill_value=0) ## bypasses self.domain.enforce_values
                return x_
            else:
                raise AttributeError(about._errors.cstring("ERROR: singular operator."))
        else:
            x_ = field(self.domain,val=None,target=x.target)
            x_.val = x.val/self.val ## bypasses self.domain.enforce_values
            return x_

    def _adjoint_inverse_multiply(self,x,pseudo=False,**kwargs): ## > applies the inverse adjoint operator to a given field
        if(np.any(self.val==0)):
            if(pseudo):
                x_ = field(self.domain,val=None,target=x.target)
                x_.val = np.ma.filled(x.val/np.ma.masked_where(self.val==0,np.conjugate(self.val),copy=False),fill_value=0) ## bypasses self.domain.enforce_values
                return x_
            else:
                raise AttributeError(about._errors.cstring("ERROR: singular operator."))
        else:
            x_ = field(self.target,val=None,target=x.target)
            x_.val = x.val/np.conjugate(self.val) ## bypasses self.domain.enforce_values
            return x_

    def _inverse_adjoint_multiply(self,x,pseudo=False,**kwargs): ## > applies the adjoint inverse operator to a given field
        if(np.any(self.val==0)):
            if(pseudo):
                x_ = field(self.domain,val=None,target=x.target)
                x_.val = np.ma.filled(x.val/np.conjugate(np.ma.masked_where(self.val==0,self.val,copy=False)),fill_value=0) ## bypasses self.domain.enforce_values
                return x_
            else:
                raise AttributeError(about._errors.cstring("ERROR: singular operator."))

        else:
            x_ = field(self.target,val=None,target=x.target)
            x_.val = x.val*np.conjugate(1/self.val) ## bypasses self.domain.enforce_values
            return x_

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def tr(self,domain=None,**kwargs):
        """
            Computes the trace of the operator

            Parameters
            ----------
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: 1)
            var : bool, *optional*
                Indicates whether to additionally return the probing variance
                or not (default: False).
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            tr : float
                Trace of the operator
            delta : float, *optional*
                Probing variance of the trace. Returned if `var` is True in
                of probing case.

        """
        if(domain is None)or(domain==self.domain):
            if(self.uni): ## identity
                return (self.domain.datatype(self.domain.dof())).real
            elif(self.domain.dim(split=False)<self.domain.dof()): ## hidden degrees of freedom
                return self.domain.calc_dot(np.ones(self.domain.dim(split=True),dtype=self.domain.datatype,order='C'),self.val) ## discrete inner product
            else:
                return np.sum(self.val,axis=None,dtype=None,out=None)
        else:
            if(self.uni): ## identity
                if(not isinstance(domain,space)):
                    raise TypeError(about._errors.cstring("ERROR: invalid input."))
                ## check degrees of freedom
                if(self.domain.dof()>domain.dof()):
                    about.infos.cprint("INFO: variant numbers of degrees of freedom ( "+str(self.domain.dof())+" / "+str(domain.dof())+" ).")
                return (domain.datatype(domain.dof())).real
            else:
                return super(diagonal_operator,self).tr(domain=domain,**kwargs) ## probing

    def inverse_tr(self,domain=None,**kwargs):
        """
            Computes the trace of the inverse operator

            Parameters
            ----------
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: 1)
            var : bool, *optional*
                Indicates whether to additionally return the probing variance
                or not (default: False).
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            tr : float
                Trace of the inverse operator
            delta : float, *optional*
                Probing variance of the trace. Returned if `var` is True in
                of probing case.

        """
        if(np.any(self.val==0)):
            raise AttributeError(about._errors.cstring("ERROR: singular operator."))

        if(domain is None)or(domain==self.target):
            if(self.uni): ## identity
                return np.real(self.domain.datatype(self.domain.dof()))
            elif(self.domain.dim(split=False)<self.domain.dof()): ## hidden degrees of freedom
                return self.domain.calc_dot(np.ones(self.domain.dim(split=True),dtype=self.domain.datatype,order='C'),1/self.val) ## discrete inner product
            else:
                return np.sum(1/self.val,axis=None,dtype=None,out=None)
        else:
            if(self.uni): ## identity
                if(not isinstance(domain,space)):
                    raise TypeError(about._errors.cstring("ERROR: invalid input."))
                ## check degrees of freedom
                if(self.domain.dof()>domain.dof()):
                    about.infos.cprint("INFO: variant numbers of degrees of freedom ( "+str(self.domain.dof())+" / "+str(domain.dof())+" ).")
                return np.real(domain.datatype(domain.dof()))
            else:
                return super(diagonal_operator,self).inverse_tr(domain=domain,**kwargs) ## probing

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def diag(self,bare=False,domain=None,**kwargs):
        """
            Computes the diagonal of the operator.

            Parameters
            ----------
            bare : bool, *optional*
                Indicatese whether the diagonal entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: False)
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: 1)
            var : bool, *optional*
                Indicates whether to additionally return the probing variance
                or not (default: False).
            save : bool, *optional*
                whether all individual probing results are saved or not
                (default: False)
            path : string, *optional*
                path wherein the results are saved (default: "tmp")
            prefix : string, *optional*
                prefix for all saved files (default: "")
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            diag : ndarray
                The matrix diagonal
            delta : float, *optional*
                Probing variance of the trace. Returned if `var` is True in
                of probing case.

            Notes
            -----
            The ambiguity of `bare` or non-bare diagonal entries is based
            on the choice of a matrix representation of the operator in
            question. The naive choice of absorbing the volume weights
            into the matrix leads to a matrix-vector calculus with the
            non-bare entries which seems intuitive, though. The choice of
            keeping matrix entries and volume weights separate deals with the
            bare entries that allow for correct interpretation of the matrix
            entries; e.g., as variance in case of an covariance operator.

        """
        if(domain is None)or(domain==self.domain):
            ## weight if ...
            if(not self.domain.discrete)and(bare):
                diag = self.domain.calc_weight(self.val,power=-1)
                ## check complexity
                if(np.all(np.imag(diag)==0)):
                    diag = np.real(diag)
                return diag
            else:
                return self.val
        else:
            if(self.uni): ## identity
                if(not isinstance(domain,space)):
                    raise TypeError(about._errors.cstring("ERROR: invalid input."))
                ## weight if ...
                if(not domain.discrete)and(bare):
                    return np.real(domain.calc_weight(domain.enforce_values(1,extend=True),power=-1))
                else:
                    return np.real(domain.enforce_values(1,extend=True))
            else:
                return super(diagonal_operator,self).diag(bare=bare,domain=domain,**kwargs) ## probing

    def inverse_diag(self,bare=False,domain=None,**kwargs):
        """
            Computes the diagonal of the inverse operator.

            Parameters
            ----------
            bare : bool, *optional*
                Indicatese whether the diagonal entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: False)
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: 1)
            var : bool, *optional*
                Indicates whether to additionally return the probing variance
                or not (default: False).
            save : bool, *optional*
                whether all individual probing results are saved or not
                (default: False)
            path : string, *optional*
                path wherein the results are saved (default: "tmp")
            prefix : string, *optional*
                prefix for all saved files (default: "")
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            diag : ndarray
                The diagonal of the inverse matrix
            delta : float, *optional*
                Probing variance of the trace. Returned if `var` is True in
                of probing case.

            See Also
            --------
            probing : The class used to perform probing operations

            Notes
            -----
            The ambiguity of `bare` or non-bare diagonal entries is based
            on the choice of a matrix representation of the operator in
            question. The naive choice of absorbing the volume weights
            into the matrix leads to a matrix-vector calculus with the
            non-bare entries which seems intuitive, though. The choice of
            keeping matrix entries and volume weights separate deals with the
            bare entries that allow for correct interpretation of the matrix
            entries; e.g., as variance in case of an covariance operator.

        """
        if(domain is None)or(domain==self.target):
            ## weight if ...
            if(not self.domain.discrete)and(bare):
                diag = self.domain.calc_weight(1/self.val,power=-1)
                ## check complexity
                if(np.all(np.imag(diag)==0)):
                    diag = np.real(diag)
                return diag
            else:
                return 1/self.val
        else:
            if(self.uni): ## identity
                if(not isinstance(domain,space)):
                    raise TypeError(about._errors.cstring("ERROR: invalid input."))
                ## weight if ...
                if(not domain.discrete)and(bare):
                    return np.real(domain.calc_weight(domain.enforce_values(1,extend=True),power=-1))
                else:
                    return np.real(domain.enforce_values(1,extend=True))
            else:
                return super(diagonal_operator,self).inverse_diag(bare=bare,domain=domain,**kwargs) ## probing

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def det(self):
        """
            Computes the determinant of the matrix.

            Returns
            -------
            det : float
                The determinant

        """
        if(self.uni): ## identity
            return 1
        elif(self.domain.dim(split=False)<self.domain.dof()): ## hidden degrees of freedom
            return np.exp(self.domain.calc_dot(np.ones(self.domain.dim(split=True),dtype=self.domain.datatype,order='C'),np.log(self.val)))
        else:
            return np.prod(self.val,axis=None,dtype=None,out=None)

    def inverse_det(self):
        """
            Computes the determinant of the inverse operator.

            Returns
            -------
            det : float
                The determinant

        """
        if(self.uni): ## identity
            return 1
        det = self.det()
        if(det<>0):
            return 1/det
        else:
            raise ValueError(about._errors.cstring("ERROR: singular operator."))

    def log_det(self):
        """
            Computes the logarithm of the determinant of the operator.

            Returns
            -------
            logdet : float
                The logarithm of the determinant

        """
        if(self.uni): ## identity
            return 0
        elif(self.domain.dim(split=False)<self.domain.dof()): ## hidden degrees of freedom
            return self.domain.calc_dot(np.ones(self.domain.dim(split=True),dtype=self.domain.datatype,order='C'),np.log(self.val))
        else:
            return np.sum(np.log(self.val),axis=None,dtype=None,out=None)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_random_field(self,domain=None,target=None):
        """
            Generates a Gaussian random field with variance equal to the
            diagonal.

            Parameters
            ----------
            domain : space, *optional*
                space wherein the field lives (default: None, indicates
                to use self.domain)
            target : space, *optional*
                space wherein the transform of the field lives
                (default: None, indicates to use target of domain)

            Returns
            -------
            x : field
                Random field.

        """
        ## weight if ...
        if(not self.domain.discrete):
            diag = self.domain.calc_weight(self.val,power=-1)
            ## check complexity
            if(np.all(np.imag(diag)==0)):
                diag = np.real(diag)
        else:
            diag = self.val

        if(domain is None)or(domain==self.domain):
            return field(self.domain,val=None,target=target,random="gau",var=self.diag(bare=True,domain=self.domain))
        else:
            return field(self.domain,val=None,target=domain,random="gau",var=self.diag(bare=True,domain=self.domain)).transform(target=domain,overwrite=False)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty_core.diagonal_operator>"

##-----------------------------------------------------------------------------

##-----------------------------------------------------------------------------

def identity(domain):
    """
        Returns an identity operator.

        The identity operator is represented by a `diagonal_operator` instance,
        which is applicable to a field-like object; i.e., a scalar, list,
        array or field. (The identity operator is unrelated to PYTHON's
        built-in function :py:func:`id`.)

        Parameters
        ----------
        domain : space
            The space wherein valid arguments live.

        Returns
        -------
        id : diagonal_operator
            The identity operator as a `diagonal_operator` instance.

        See Also
        --------
        diagonal_operator

        Examples
        --------
        >>> I = identity(rg_space(8,dist=0.2))
        >>> I.diag()
        array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
        >>> I.diag(bare=True)
        array([ 5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.])
        >>> I.tr()
        8.0
        >>> I(3)
        <nifty.field>
        >>> I(3).val
        array([ 3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.])
        >>> I(np.arange(8))[:]
        array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.])
        >>> f = I.get_random_field()
        >>> print(I(f) - f)
        nifty.field instance
        - domain      = <nifty.rg_space>
        - val         = [...]
          - min.,max. = [0.0, 0.0]
          - med.,mean = [0.0, 0.0]
        - target      = <nifty.rg_space>
        >>> I.times(f) ## equal to I(f)
        <nifty.field>
        >>> I.inverse_times(f)
        <nifty.field>

    """
    return diagonal_operator(domain=domain,diag=1,bare=False)

##-----------------------------------------------------------------------------

##-----------------------------------------------------------------------------

class power_operator(diagonal_operator):
    """
        ..      ______    ______   __     __   _______   _____
        ..    /   _   | /   _   | |  |/\/  / /   __  / /   __/
        ..   /  /_/  / /  /_/  /  |       / /  /____/ /  /
        ..  /   ____/  \______/   |__/\__/  \______/ /__/     operator class
        .. /__/

        NIFTY subclass for (signal-covariance-type) diagonal operators containing a power spectrum

        Parameters
        ----------
        domain : space, *optional*
            The space wherein valid arguments live. If no domain is given
            then the diag parameter *must* be a field and the domain
            of that field is assumed. (default: None)
        spec : {scalar, list, array, field, function}
            The power spectrum. For a scalar, a constant power
            spectrum is defined having the value provided. If no domain
            is given, diag must be a field. (default: 1)
        bare : bool, *optional*
            whether the entries are `bare` or not
            (mandatory for the correct incorporation of volume weights)
            (default: True)
        pindex : ndarray, *optional*
            indexing array, obtainable from domain.get_power_indices
            (default: None)

        Other Parameters
        ----------------
        log : bool, *optional*
            Flag specifying if the spectral binning is performed on logarithmic
            scale or not; if set, the number of used bins is set
            automatically (if not given otherwise); by default no binning
            is done (default: None).
        nbin : integer, *optional*
            Number of used spectral bins; if given `log` is set to ``False``;
            integers below the minimum of 3 induce an automatic setting;
            by default no binning is done (default: None).
        binbounds : {list, array}, *optional*
            User specific inner boundaries of the bins, which are preferred
            over the above parameters; by default no binning is done
            (default: None).            vmin : {scalar, list, ndarray, field}, *optional*
            Lower limit of the uniform distribution if ``random == "uni"``
            (default: 0).

        Notes
        -----
        The ambiguity of `bare` or non-bare diagonal entries is based
        on the choice of a matrix representation of the operator in
        question. The naive choice of absorbing the volume weights
        into the matrix leads to a matrix-vector calculus with the
        non-bare entries which seems intuitive, though. The choice of
        keeping matrix entries and volume weights separate deals with the
        bare entries that allow for correct interpretation of the matrix
        entries; e.g., as variance in case of an covariance operator.

        Attributes
        ----------
        domain : space
            The space wherein valid arguments live.
        val : ndarray
            A field containing the diagonal entries of the matrix.
        sym : bool
            Indicates whether the operator is self-adjoint or not
        uni : bool
            Indicates whether the operator is unitary or not
        imp : bool
            Indicates whether the incorporation of volume weights in
            multiplications is already implemented in the `multiply`
            instance methods or not
        target : space
            The space wherein the operator output lives

    """
    def __init__(self,domain,spec=1,bare=True,pindex=None,**kwargs):
        """
            Sets the diagonal operator's standard properties

            Parameters
            ----------
            domain : space, *optional*
                The space wherein valid arguments live. If no domain is given
                then the diag parameter *must* be a field and the domain
                of that field is assumed. (default: None)
            spec : {scalar, list, array, field, function}
                The power spectrum. For a scalar, a constant power
                spectrum is defined having the value provided. If no domain
                is given, diag must be a field. (default: 1)
            bare : bool, *optional*
                whether the entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: True)
            pindex : ndarray, *optional*
                indexing array, obtainable from domain.get_power_indices
                (default: None)

            Returns
            -------
            None

            Other Parameters
            ----------------
            log : bool, *optional*
                Flag specifying if the spectral binning is performed on logarithmic
                scale or not; if set, the number of used bins is set
                automatically (if not given otherwise); by default no binning
                is done (default: None).
            nbin : integer, *optional*
                Number of used spectral bins; if given `log` is set to ``False``;
                integers below the minimum of 3 induce an automatic setting;
                by default no binning is done (default: None).
            binbounds : {list, array}, *optional*
                User specific inner boundaries of the bins, which are preferred
                over the above parameters; by default no binning is done
                (default: None).            vmin : {scalar, list, ndarray, field}, *optional*
                Lower limit of the uniform distribution if ``random == "uni"``
                (default: 0).

        """
        if(not isinstance(domain,space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        self.domain = domain
        ## check implicit pindex
        if(pindex is None):
            try:
                self.domain.set_power_indices(**kwargs)
            except:
                raise ValueError(about._errors.cstring("ERROR: invalid input."))
            else:
                pindex = self.domain.power_indices.get("pindex")
        ## check explicit pindex
        else:
            pindex = np.array(pindex,dtype=np.int)
            if(not np.all(np.array(np.shape(pindex))==self.domain.dim(split=True))):
                raise ValueError(about._errors.cstring("ERROR: shape mismatch ( "+str(np.array(np.shape(pindex)))+" <> "+str(self.domain.dim(split=True))+" )."))
        ## set diagonal
        try:
            #diag = self.domain.enforce_power(spec,size=np.max(pindex,axis=None,out=None)+1)[pindex]
            temp_spec = self.domain.enforce_power(
                            spec,size=np.max(pindex,axis=None,out=None)+1)            
            diag = pindex.apply_scalar_function(lambda x: temp_spec[x])
        except(AttributeError):
            raise ValueError(about._errors.cstring("ERROR: invalid input."))
        ## weight if ...
        if(not self.domain.discrete)and(bare):
            self.val = np.real(self.domain.calc_weight(diag,power=1))
        else:
            self.val = diag

        self.sym = True

        ## check whether identity
        if(np.all(spec==1)):
            self.uni = True
        else:
            self.uni = False

        self.imp = True
        self.target = self.domain

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def set_power(self,newspec,bare=True,pindex=None,**kwargs):
        """
            Sets the power spectrum of the diagonal operator

            Parameters
            ----------
            newspec : {scalar, list, array, field, function}
                The entries of the operator. For a scalar, a constant
                diagonal is defined having the value provided. If no domain
                is given, diag must be a field. (default: 1)
            bare : bool
                whether the entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
            pindex : ndarray, *optional*
                indexing array, obtainable from domain.get_power_indices
                (default: None)

            Returns
            -------
            None

            Other Parameters
            ----------------
            log : bool, *optional*
                Flag specifying if the spectral binning is performed on logarithmic
                scale or not; if set, the number of used bins is set
                automatically (if not given otherwise); by default no binning
                is done (default: None).
            nbin : integer, *optional*
                Number of used spectral bins; if given `log` is set to ``False``;
                integers below the minimum of 3 induce an automatic setting;
                by default no binning is done (default: None).
            binbounds : {list, array}, *optional*
                User specific inner boundaries of the bins, which are preferred
                over the above parameters; by default no binning is done
                (default: None).

        """
#        if(bare is None):
#            about.warnings.cprint("WARNING: bare keyword set to default.")
#            bare = True
        ## check implicit pindex
        if(pindex is None):
            try:
                self.domain.set_power_indices(**kwargs)
            except:
                raise ValueError(about._errors.cstring("ERROR: invalid domain."))
            else:
                pindex = self.domain.power_indices.get("pindex")
        ## check explicit pindex
        else:
            pindex = np.array(pindex,dtype=np.int)
            if(not np.all(np.array(np.shape(pindex))==self.domain.dim(split=True))):
                raise ValueError(about._errors.cstring("ERROR: shape mismatch ( "+str(np.array(np.shape(pindex)))+" <> "+str(self.domain.dim(split=True))+" )."))
        ## set diagonal
        try:
            diag = self.domain.enforce_power(newspec,size=np.max(pindex,axis=None,out=None)+1)[pindex]
        except(AttributeError):
            raise ValueError(about._errors.cstring("ERROR: invalid input."))
        ## weight if ...
        if(not self.domain.discrete)and(bare):
            self.val = np.real(self.domain.calc_weight(diag,power=1))
        else:
            self.val = diag

        ## check whether identity
        if(np.all(newspec==1)):
            self.uni = True
        else:
            self.uni = False

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_power(self,bare=True,pundex=None,pindex=None,**kwargs):
        """
            Computes the power spectrum

            Parameters
            ----------
            bare : bool, *optional*
                whether the entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: True)
            pundex : ndarray, *optional*
                unindexing array, obtainable from domain.get_power_indices
                (default: None)
            pindex : ndarray, *optional*
                indexing array, obtainable from domain.get_power_indices
                (default: None)

            Returns
            -------
            spec : ndarray
                The power spectrum

            Other Parameters
            ----------------
            log : bool, *optional*
                Flag specifying if the spectral binning is performed on logarithmic
                scale or not; if set, the number of used bins is set
                automatically (if not given otherwise); by default no binning
                is done (default: None).
            nbin : integer, *optional*
                Number of used spectral bins; if given `log` is set to ``False``;
                integers below the minimum of 3 induce an automatic setting;
                by default no binning is done (default: None).
            binbounds : {list, array}, *optional*
                User specific inner boundaries of the bins, which are preferred
                over the above parameters; by default no binning is done
                (default: None).            vmin : {scalar, list, ndarray, field}, *optional*
                Lower limit of the uniform distribution if ``random == "uni"``
                (default: 0).

        """
        ## weight if ...
        if(not self.domain.discrete)and(bare):
            diag = np.real(self.domain.calc_weight(self.val,power=-1))
        else:
            diag = self.val
        ## check implicit pundex
        if(pundex is None):
            if(pindex is None):
                try:
                    self.domain.set_power_indices(**kwargs)
                except:
                    raise ValueError(about._errors.cstring("ERROR: invalid domain."))
                else:
                    pundex = self.domain.power_indices.get("pundex")
            else:
                pindex = np.array(pindex,dtype=np.int)
                if(not np.all(np.array(np.shape(pindex))==self.domain.dim(split=True))):
                    raise ValueError(about._errors.cstring("ERROR: shape mismatch ( "+str(np.array(np.shape(pindex)))+" <> "+str(self.domain.dim(split=True))+" )."))
                ## quick pundex
                pundex = np.unique(pindex,return_index=True,return_inverse=False)[1]
        ## check explicit pundex
        else:
            pundex = np.array(pundex,dtype=np.int)

        return diag.flatten(order='C')[pundex]

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_projection_operator(self,pindex=None,**kwargs):
        """
            Generates a spectral projection operator

            Parameters
            ----------
            pindex : ndarray
                indexing array obtainable from domain.get_power_indices
                (default: None)

            Returns
            -------
            P : projection_operator

            Other Parameters
            ----------------
            log : bool, *optional*
                Flag specifying if the spectral binning is performed on logarithmic
                scale or not; if set, the number of used bins is set
                automatically (if not given otherwise); by default no binning
                is done (default: None).
            nbin : integer, *optional*
                Number of used spectral bins; if given `log` is set to ``False``;
                integers below the minimum of 3 induce an automatic setting;
                by default no binning is done (default: None).
            binbounds : {list, array}, *optional*
                User specific inner boundaries of the bins, which are preferred
                over the above parameters; by default no binning is done
                (default: None).            vmin : {scalar, list, ndarray, field}, *optional*
                Lower limit of the uniform distribution if ``random == "uni"``
                (default: 0).

        """
        ## check implicit pindex
        if(pindex is None):
            try:
                self.domain.set_power_indices(**kwargs)
            except:
                raise ValueError(about._errors.cstring("ERROR: invalid domain."))
            else:
                pindex = self.domain.power_indices.get("pindex")
        ## check explicit pindex
        else:
            pindex = np.array(pindex,dtype=np.int)
            if(not np.all(np.array(np.shape(pindex))==self.domain.dim(split=True))):
                raise ValueError(about._errors.cstring("ERROR: shape mismatch ( "+str(np.array(np.shape(pindex)))+" <> "+str(self.domain.dim(split=True))+" )."))

        return projection_operator(self.domain,assign=pindex)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty_core.power_operator>"

##-----------------------------------------------------------------------------



##-----------------------------------------------------------------------------

class projection_operator(operator):
    """
        ..                                     __                       __     __
        ..                                   /__/                     /  /_  /__/
        ..      ______    _____   ______     __   _______   _______  /   _/  __   ______    __ ___
        ..    /   _   | /   __/ /   _   |  /  / /   __  / /   ____/ /  /   /  / /   _   | /   _   |
        ..   /  /_/  / /  /    /  /_/  /  /  / /  /____/ /  /____  /  /_  /  / /  /_/  / /  / /  /
        ..  /   ____/ /__/     \______/  /  /  \______/  \______/  \___/ /__/  \______/ /__/ /__/  operator class
        .. /__/                        /___/

        NIFTY subclass for projection operators

        Parameters
        ----------
        domain : space
            The space wherein valid arguments live.
        assign : ndarray, *optional*
            Assignments of domain items to projection bands. An array
            of integers, negative integers are associated with the
            nullspace of the projection. (default: None)

        Other Parameters
        ----------------
        log : bool, *optional*
            Flag specifying if the spectral binning is performed on logarithmic
            scale or not; if set, the number of used bins is set
            automatically (if not given otherwise); by default no binning
            is done (default: None).
        nbin : integer, *optional*
            Number of used spectral bins; if given `log` is set to ``False``;
            integers below the minimum of 3 induce an automatic setting;
            by default no binning is done (default: None).
        binbounds : {list, array}, *optional*
            User specific inner boundaries of the bins, which are preferred
            over the above parameters; by default no binning is done
            (default: None).            vmin : {scalar, list, ndarray, field}, *optional*
            Lower limit of the uniform distribution if ``random == "uni"``
            (default: 0).

        Notes
        -----
        The application of the projection operator features a ``band`` keyword
        specifying a single projection band (see examples), a ``bandsup``
        keyword specifying which projection bands to sum up, and a ``split``
        keyword.

        Examples
        --------
        >>> space = point_space(3)
        >>> P = projection_operator(space, assign=[0, 1, 0])
        >>> P.bands()
        2
        >>> P([1, 2, 3], band=0) # equal to P.times(field(space,val=[1, 2, 3]))
        <nifty.field>
        >>> P([1, 2, 3], band=0).domain
        <nifty.point_space>
        >>> P([1, 2, 3], band=0).val # projection on band 0 (items 0 and 2)
        array([ 1.,  0.,  3.])
        >>> P([1, 2, 3], band=1).val # projection on band 1 (item 1)
        array([ 0.,  2.,  0.])
        >>> P([1, 2, 3])
        <nifty.field>
        >>> P([1, 2, 3]).domain
        <nifty.nested_space>
        >>> P([1, 2, 3]).val # projection on all bands
        array([[ 1.,  0.,  3.],
               [ 0.,  2.,  0.]])

        Attributes
        ----------
        domain : space
            The space wherein valid arguments live.
        ind : ndarray
            Assignments of domain items to projection bands.
        sym : bool
            Indicates whether the operator is self-adjoint or not
        uni : bool
            Indicates whether the operator is unitary or not
        imp : bool
            Indicates whether the incorporation of volume weights in
            multiplications is already implemented in the `multiply`
            instance methods or not
        target : space
            The space wherein the operator output lives

    """
    def __init__(self,domain,assign=None,**kwargs):
        """
            Sets the standard operator properties and `indexing`.

            Parameters
            ----------
            domain : space
                The space wherein valid arguments live.
            assign : ndarray, *optional*
                Assignments of domain items to projection bands. An array
                of integers, negative integers are associated with the
                nullspace of the projection. (default: None)

            Returns
            -------
            None

            Other Parameters
            ----------------
            log : bool, *optional*
                Flag specifying if the spectral binning is performed on logarithmic
                scale or not; if set, the number of used bins is set
                automatically (if not given otherwise); by default no binning
                is done (default: None).
            nbin : integer, *optional*
                Number of used spectral bins; if given `log` is set to ``False``;
                integers below the minimum of 3 induce an automatic setting;
                by default no binning is done (default: None).
            binbounds : {list, array}, *optional*
                User specific inner boundaries of the bins, which are preferred
                over the above parameters; by default no binning is done
                (default: None).            vmin : {scalar, list, ndarray, field}, *optional*
                Lower limit of the uniform distribution if ``random == "uni"``
                (default: 0).

        """
        if(not isinstance(domain,space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        self.domain = domain

        ## check assignment(s)
        if(assign is None):
            try:
                self.domain.set_power_indices(**kwargs)
            except:
                assign = np.arange(self.domain.dim(split=False),dtype=np.int)
            else:
                assign = self.domain.power_indices.get("pindex").flatten(order='C')
        else:
            assign = self.domain.enforce_shape(assign).astype(np.int).flatten(order='C')
        ## build indexing
        self.ind = [np.where(assign==ii)[0] for ii in xrange(np.max(assign,axis=None,out=None)+1) if ii in assign]

        self.sym = True
#        about.infos.cprint("INFO: pseudo unitary projection operator.")
        self.uni = False
        self.imp = True

        self.target = nested_space([point_space(len(self.ind),datatype=self.domain.datatype),self.domain])

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def bands(self):
        """
            Computes the number of projection bands

            Returns
            -------
            bands : int
                The number of projection bands
        """
        return len(self.ind)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def rho(self):
        """
            Computes the number of degrees of freedom per projection band.

            Returns
            -------
            rho : ndarray
                The number of degrees of freedom per projection band.
        """
        rho = np.empty(len(self.ind),dtype=np.int,order='C')
        if(self.domain.dim(split=False)==self.domain.dof()): ## no hidden degrees of freedom
            for ii in xrange(len(self.ind)):
                rho[ii] = len(self.ind[ii])
        else: ## hidden degrees of freedom
            mof = np.round(np.real(self.domain.calc_weight(self.domain.get_meta_volume(total=False),power=-1).flatten(order='C')),decimals=0,out=None).astype(np.int) ## meta degrees of freedom
            for ii in xrange(len(self.ind)):
                rho[ii] = np.sum(mof[self.ind[ii]],axis=None,dtype=np.int,out=None)
        return rho

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _multiply(self,x,band=None,bandsup=None,**kwargs):
        """
            Applies the operator to a given field.

            Parameters
            ----------
            x : field
                Valid input field.
            band : int, *optional*
                Projection band whereon to project (default: None).
            bandsup: {integer, list/array of integers}, *optional*
                List of projection bands whereon to project and which to sum
                up. The `band` keyword is prefered over `bandsup`
                (default: None).

            Returns
            -------
            Px : field
                projected field(!)
        """
        if(band is not None):
            band = int(band)
            if(band>self.bands()-1)or(band<0):
                raise TypeError(about._errors.cstring("ERROR: invalid band."))
            Px = np.zeros(self.domain.dim(split=False),dtype=self.domain.datatype,order='C')
            Px[self.ind[band]] += x.val.flatten(order='C')[self.ind[band]]
            Px = field(self.domain,val=Px,target=x.target)
            return Px

        elif(bandsup is not None):
            if(np.isscalar(bandsup)):
                bandsup = np.arange(int(bandsup+1),dtype=np.int)
            else:
                bandsup = np.array(bandsup,dtype=np.int)
            if(np.any(bandsup>self.bands()-1))or(np.any(bandsup<0)):
                raise ValueError(about._errors.cstring("ERROR: invalid input."))
            Px = np.zeros(self.domain.dim(split=False),dtype=self.domain.datatype,order='C')
            x_ = x.val.flatten(order='C')
            for bb in bandsup:
                Px[self.ind[bb]] += x_[self.ind[bb]]
            Px = field(self.domain,val=Px,target=x.target)
            return Px

        else:
            Px = np.zeros((len(self.ind),self.domain.dim(split=False)),dtype=self.target.datatype,order='C')
            x_ = x.val.flatten(order='C')
            for bb in xrange(self.bands()):
                Px[bb][self.ind[bb]] += x_[self.ind[bb]]
            Px = field(self.target,val=Px,target=nested_space([point_space(len(self.ind),datatype=x.target.datatype),x.target]))
            return Px

    def _inverse_multiply(self,x,**kwargs):
        raise AttributeError(about._errors.cstring("ERROR: singular operator.")) ## pseudo unitary

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _debriefing(self,x,x_,target,inverse): ## > evaluates x and x_ after `multiply`
        if(x_ is None):
            return None
        else:
            ## weight if ...
            if(not self.imp)and(not target.discrete)and(inverse):
                x_ = x_.weight(power=-1,overwrite=False)
            ## inspect x
            if(isinstance(x,field)):
                if(x_.domain==self.target):
                    ## repair ...
                    if(x_.domain.nest[-1]!=x.domain):
                        x_ = x_.transform(target=nested_space([point_space(len(self.ind),datatype=x.domain.datatype),x.domain]),overwrite=False) ## ... domain
                    if(x_.target.nest[-1]!=x.target):
                        x_.set_target(newtarget=nested_space([point_space(len(self.ind),datatype=x.target.datatype),x.target])) ## ... codomain
                else:
                    ## repair ...
                    if(x_.domain!=x.domain):
                        x_ = x_.transform(target=x.domain,overwrite=False) ## ... domain
                    if(x_.target!=x.target):
                        x_.set_target(newtarget=x.target) ## ... codomain
            return x_

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def times(self,x,**kwargs):
        """
            Applies the operator to a given object

            Parameters
            ----------
            x : {scalar, ndarray, field}
                Scalars are interpreted as constant arrays, and an array will
                be interpreted as a field on the domain of the operator.

            Returns
            -------
            Ox : {field, tuple of fields}
                Mapped field on the target domain of the operator.

            Other Parameters
            ----------------
            band : int, *optional*
                Projection band whereon to project (default: None).
            bandsup: {integer, list/array of integers}, *optional*
                List of projection bands whereon to project and which to sum
                up. The `band` keyword is prefered over `bandsup`
                (default: None).
            split: bool, *optional*
                Whether to return a tuple of the projected and residual field;
                applys only if `band` or `bandsup` is given
                (default: False).

        """
        ## prepare
        x_ = self._briefing(x,self.domain,False)
        ## apply operator
        x_ = self._multiply(x_,**kwargs)
        ## evaluate
        y = self._debriefing(x,x_,self.target,False)
        ## split if ...
        if(kwargs.get("split",False))and((kwargs.has_key("band"))or(kwargs.has_key("bandsup"))):
            return y,x-y
        else:
            return y

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def pseudo_tr(self,x,**kwargs):
        """
            Computes the pseudo trace of a given object for all projection bands

            Parameters
            ----------
            x : {field, operator}
                The object whose pseudo-trace is to be computed. If the input is
                a field, the pseudo trace equals the trace of
                the projection operator mutliplied by a vector-vector operator
                corresponding to the input field. This is also equal to the
                pseudo inner product of the field with projected field itself.
                If the input is a operator, the pseudo trace equals the trace of
                the projection operator multiplied by the input operator.
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: 1)
            save : bool, *optional*
                whether all individual probing results are saved or not
                (default: False)
            path : string, *optional*
                path wherein the results are saved (default: "tmp")
            prefix : string, *optional*
                prefix for all saved files (default: "")
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            tr : float
                Pseudo trace for all projection bands
        """
        if(isinstance(x,operator)):
            ## compute non-bare diagonal of the operator x
            x = x.diag(bare=False,domain=self.domain,target=x.domain,var=False,**kwargs)
            if(x is None):
                raise TypeError(about._error.cstring("ERROR: 'NoneType' encountered."))

        elif(isinstance(x,field)):
            ## check domain
            if(self.domain==x.domain):
                x = x.val
            else:
                x = x.transform(target=self.domain,overwrite=False).val
            ## compute non-bare diagonal of the vector-vector operator corresponding to the field x
            x = x*np.conjugate(x)
            ## weight
            if(not self.domain.discrete):
                x = self.domain.calc_weight(x,power=1)

        else:
            raise TypeError(about._errors.cstring("ERROR: invalid input."))

        x = np.real(x.flatten(order='C'))
        if(not self.domain.dim(split=False)==self.domain.dof()):
            x *= np.round(np.real(self.domain.calc_weight(self.domain.get_meta_volume(total=False),power=-1).flatten(order='C')),decimals=0,out=None).astype(np.int) ## meta degrees of freedom

        tr = np.empty(self.bands(),dtype=x.dtype,order='C')
        for bb in xrange(self.bands()):
            tr[bb] = np.sum(x[self.ind[bb]],axis=None,dtype=None,out=None)
        return tr

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty_core.projection_operator>"

##-----------------------------------------------------------------------------



##-----------------------------------------------------------------------------

class vecvec_operator(operator):
    """
        ..                                                                 __
        ..                                                             __/  /__
        ..  __   __   _______   _______  __   __   _______   _______ /__    __/
        .. |  |/  / /   __  / /   ____/ |  |/  / /   __  / /   ____/   /__/
        .. |     / /  /____/ /  /____   |     / /  /____/ /  /____
        .. |____/  \______/  \______/   |____/  \______/  \______/            operator class

        NIFTY subclass for vector-vector operators

        Parameters
        ----------
        domain : space, *optional*
            The space wherein valid arguments live. If none is given, the
            space of the field given in val is used. (default: None)
        val : {scalar, ndarray, field}, *optional*
            The field from which to construct the operator. For a scalar, a constant
            field is defined having the value provided. If no domain
            is given, val must be a field. (default: 1)

        Attributes
        ----------
        domain : space
            The space wherein valid arguments live.
        val : ndarray
            The field from which the operator is derived.
        sym : bool
            Indicates whether the operator is self-adjoint or not
        uni : bool
            Indicates whether the operator is unitary or not
        imp : bool
            Indicates whether the incorporation of volume weights in
            multiplications is already implemented in the `multiply`
            instance methods or not
        target : space
            The space wherein the operator output lives.
    """
    def __init__(self,domain=None,val=1):
        """
            Sets the standard operator properties and `values`.

            Parameters
            ----------
            domain : space, *optional*
                The space wherein valid arguments live. If none is given, the
                space of the field given in val is used. (default: None)
            val : {scalar, ndarray, field}, *optional*
                The field from which to construct the operator. For a scalar, a constant
                field is defined having the value provided. If no domain
                is given, val must be a field. (default: 1)

            Returns
            -------
            None
        """
        if(domain is None)and(isinstance(val,field)):
            domain = val.domain
        if(not isinstance(domain,space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        self.domain = domain
        self.val = self.domain.enforce_values(val,extend=True)
        self.sym = True
        self.uni = False
        self.imp = False
        self.target = self.domain

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def set_val(self,newval):
        """
            Sets the field values of the operator

            Parameters
            ----------
            newval : {scalar, ndarray, field}
                The new field values. For a scalar, a constant
                diagonal is defined having the value provided. If no domain
                is given, diag must be a field. (default: 1)

            Returns
            -------
            None
        """
        self.val = self.domain.enforce_values(newval,extend=True)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _multiply(self,x,**kwargs): ## > applies the operator to a given field
        x_ = field(self.target,val=None,target=x.target)
        x_.val = self.val*self.domain.calc_dot(self.val,x.val) ## bypasses self.domain.enforce_values
        return x_

    def _inverse_multiply(self,x,**kwargs):
        raise AttributeError(about._errors.cstring("ERROR: singular operator."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def tr(self,domain=None,**kwargs):
        """
            Computes the trace of the operator

            Parameters
            ----------
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: 1)
            var : bool, *optional*
                Indicates whether to additionally return the probing variance
                or not (default: False).
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            tr : float
                Trace of the operator
            delta : float, *optional*
                Probing variance of the trace. Returned if `var` is True in
                of probing case.

        """
        if(domain is None)or(domain==self.domain):
            if(not self.domain.discrete):
                return self.domain.calc_dot(self.val,self.domain.calc_weight(self.val,power=1))
            else:
                return self.domain.calc_dot(self.val,self.val)
        else:
            return super(vecvec_operator,self).tr(domain=domain,**kwargs) ## probing

    def inverse_tr(self):
        """
        Inverse is ill-defined for this operator.
        """
        raise AttributeError(about._errors.cstring("ERROR: singular operator."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def diag(self,bare=False,domain=None,**kwargs):
        """
            Computes the diagonal of the operator.

            Parameters
            ----------
            bare : bool, *optional*
                Indicatese whether the diagonal entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: False)
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: 1)
            var : bool, *optional*
                Indicates whether to additionally return the probing variance
                or not (default: False).
            save : bool, *optional*
                whether all individual probing results are saved or not
                (default: False)
            path : string, *optional*
                path wherein the results are saved (default: "tmp")
            prefix : string, *optional*
                prefix for all saved files (default: "")
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            diag : ndarray
                The matrix diagonal
            delta : float, *optional*
                Probing variance of the trace. Returned if `var` is True in
                of probing case.

            Notes
            -----
            The ambiguity of `bare` or non-bare diagonal entries is based
            on the choice of a matrix representation of the operator in
            question. The naive choice of absorbing the volume weights
            into the matrix leads to a matrix-vector calculus with the
            non-bare entries which seems intuitive, though. The choice of
            keeping matrix entries and volume weights separate deals with the
            bare entries that allow for correct interpretation of the matrix
            entries; e.g., as variance in case of an covariance operator.

        """
        if(domain is None)or(domain==self.domain):
            diag = np.real(self.val*np.conjugate(self.val)) ## bare diagonal
            ## weight if ...
            if(not self.domain.discrete)and(not bare):
                return self.domain.calc_weight(diag,power=1)
            else:
                return diag
        else:
            return super(vecvec_operator,self).diag(bare=bare,domain=domain,**kwargs) ## probing

    def inverse_diag(self):
        """
            Inverse is ill-defined for this operator.

        """
        raise AttributeError(about._errors.cstring("ERROR: singular operator."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def det(self):
        """
            Computes the determinant of the operator.

            Returns
            -------
            det : 0
                The determinant

        """
        return 0

    def inverse_det(self):
        """
            Inverse is ill-defined for this operator.

        """
        raise AttributeError(about._errors.cstring("ERROR: singular operator."))

    def log_det(self):
        """
            Logarithm of the determinant is ill-defined for this singular operator.

        """
        raise AttributeError(about._errors.cstring("ERROR: singular operator."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty_core.vecvec_operator>"

##-----------------------------------------------------------------------------



##-----------------------------------------------------------------------------

class response_operator(operator):
    """
        ..     _____   _______   _______   ______    ______    __ ___    _______   _______
        ..   /   __/ /   __  / /  _____/ /   _   | /   _   | /   _   | /  _____/ /   __  /
        ..  /  /    /  /____/ /_____  / /  /_/  / /  /_/  / /  / /  / /_____  / /  /____/
        .. /__/     \______/ /_______/ /   ____/  \______/ /__/ /__/ /_______/  \______/  operator class
        ..                            /__/

        NIFTY subclass for response operators (of a certain family)

        Any response operator handles Gaussian convolutions, itemwise masking,
        and selective mappings.

        Parameters
        ----------
        domain : space
            The space wherein valid arguments live.
        sigma : float, *optional*
            The standard deviation of the Gaussian kernel. Zero indicates
            no convolution. (default: 0)
        mask : {scalar, ndarray}, *optional*
            Masking values for arguments (default: 1)
        assign : {list, ndarray}, *optional*
            Assignments of codomain items to domain items. A list of
            indices/ index tuples or a one/ two-dimensional array.
            (default: None)
        den : bool, *optional*
            Whether to consider the arguments as densities or not.
            Mandatory for the correct incorporation of volume weights.
            (default: False)
        target : space, *optional*
            The space wherein the operator output lives (default: domain)

        Attributes
        ----------
        domain : space
            The space wherein valid arguments live.
        sym : bool
            Indicates whether the operator is self-adjoint or not.
        uni : bool
            Indicates whether the operator is unitary or not.
        imp : bool
            Indicates whether the incorporation of volume weights in
            multiplications is already implemented in the `multiply`
            instance methods or not.
        target : space
            The space wherein the operator output lives
        sigma : float
            The standard deviation of the Gaussian kernel. Zero indicates
            no convolution.
        mask : {scalar, ndarray}
            Masking values for arguments
        assign : {list, ndarray}
            Assignments of codomain items to domain items. A list of
            indices/ index tuples or a one/ two-dimensional array.
        den : bool
            Whether to consider the arguments as densities or not.
            Mandatory for the correct incorporation of volume weights.
    """
    def __init__(self,domain,sigma=0,mask=1,assign=None,den=False,target=None):
        """
            Sets the standard properties and `density`, `sigma`, `mask` and `assignment(s)`.

            Parameters
            ----------
            domain : space
                The space wherein valid arguments live.
            sigma : float, *optional*
                The standard deviation of the Gaussian kernel. Zero indicates
                no convolution. (default: 0)
            mask : {scalar, ndarray}, *optional*
                Masking values for arguments (default: 1)
            assign : {list, ndarray}, *optional*
                Assignments of codomain items to domain items. A list of
                indices/ index tuples or a one/ two-dimensional array.
                (default: None)
            den : bool, *optional*
                Whether to consider the arguments as densities or not.
                Mandatory for the correct incorporation of volume weights.
                (default: False)
            target : space, *optional*
                The space wherein the operator output lives (default: domain)

            Returns
            -------
            None
        """
        if(not isinstance(domain,space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        self.domain = domain

        self.sym = False
        self.uni = False
        self.imp = False
        self.den = bool(den)

        self.mask = self.domain.enforce_values(mask,extend=False)

        ## check sigma
        if(sigma<0):
            raise ValueError(about._errors.cstring("ERROR: invalid sigma."))
        self.sigma = sigma

        ## check assignment(s)
        if(assign is None):
            ## 1:1 assignment
            assignments = self.domain.dim(split=False)
            self.assign = None
        elif(np.size(self.domain.dim(split=True))==1):
            if(np.isscalar(assign)):
                ## X:1 assignment
                assignments = 1
                if(assign[0]>=self.domain.dim(split=False))or(assign[0]<-self.domain.dim(split=False)):
                    raise IndexError(about._errors.cstring("ERROR: invalid bounds."))
                self.assign = [int(assign)]
            else:
                assign = np.array(assign,dtype=np.int)
                assignments = len(assign)
                if(np.ndim(assign)!=1):
                    raise ValueError(about._errors.cstring("ERROR: invalid input."))
                elif(np.any(assign>=self.domain.dim(split=False)))or(np.any(assign<-self.domain.dim(split=False))):
                    raise IndexError(about._errors.cstring("ERROR: invalid bounds."))
                if(assignments==len(np.unique(assign,return_index=False,return_inverse=False))):
                    self.assign = assign.tolist()
                else:
                    self.assign = assign
        else:
            if(np.isscalar(assign)):
                raise ValueError(about._errors.cstring("ERROR: invalid input."))
            else:
                assign = np.array(assign,dtype=np.int)
                assignments = np.size(assign,axis=0)
                if(np.ndim(assign)!=2)or(np.size(assign,axis=1)!=np.size(self.domain.dim(split=True))):
                    raise ValueError(about._errors.cstring("ERROR: invalid input."))
                for ii in xrange(np.size(assign,axis=1)):
                    if(np.any(assign[:,ii]>=self.domain.dim(split=True)[ii]))or(np.any(assign[:,ii]<-self.domain.dim(split=True)[ii])):
                        raise IndexError(about._errors.cstring("ERROR: invalid bounds."))
                if(assignments==len(np.unique(np.ravel_multi_index(assign.T,self.domain.dim(split=True),mode="raise",order='C'),return_index=False,return_inverse=False))):
                    self.assign = assign.T.tolist()
                else:
                    self.assign = assign

        if(target is None):
            ## set target
            target = point_space(assignments,datatype=self.domain.datatype)
        else:
            ## check target
            if(not isinstance(target,space)):
                raise TypeError(about._errors.cstring("ERROR: invalid input."))
            elif(not target.discrete):
                raise ValueError(about._errors.cstring("ERROR: continuous codomain.")) ## discrete(!)
            elif(np.size(target.dim(split=True))!=1):
                raise ValueError(about._errors.cstring("ERROR: structured codomain.")) ## unstructured(!)
            elif(assignments!=target.dim(split=False)):
                raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( "+str(assignments)+" <> "+str(target.dim(split=False))+" )."))
        self.target = target

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def set_sigma(self,newsigma):
        """
            Sets the standard deviation of the response operator, indicating
            the amount of convolution.

            Parameters
            ----------
            sigma : float
                The standard deviation of the Gaussian kernel. Zero indicates
                no convolution.

            Returns
            -------
            None
        """
        ## check sigma
        if(newsigma<0):
            raise ValueError(about._errors.cstring("ERROR: invalid sigma."))
        self.sigma = newsigma

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def set_mask(self,newmask):
        """
            Sets the masking values of the response operator

            Parameters
            ----------
            newmask : {scalar, ndarray}
                masking values for arguments

            Returns
            -------
            None
        """
        self.mask = self.domain.enforce_values(newmask,extend=False)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _multiply(self,x,**kwargs): ## > applies the operator to a given field
        ## smooth
        x_ = self.domain.calc_smooth(x.val,sigma=self.sigma)
        ## mask
        x_ *= self.mask
        ## assign and return
        if(self.assign is None):
            return field(self.target,val=x_,target=kwargs.get("target",None))
        elif(isinstance(self.assign,list)):
            return field(self.target,val=x_[self.assign],target=kwargs.get("target",None))
        else:
            return field(self.target,val=x_[self.assign.T.tolist()],target=kwargs.get("target",None))

    def _adjoint_multiply(self,x,**kwargs): ## > applies the adjoint operator to a given field
        x_ = np.zeros(self.domain.dim(split=True),dtype=self.domain.datatype,order='C')
        ## assign (transposed)
        if(self.assign is None):
            x_ = np.copy(x.val.flatten(order='C'))
        elif(isinstance(self.assign,list)):
            x_[self.assign] += x.val.flatten(order='C')
        else:
            for ii in xrange(np.size(self.assign,axis=0)):
                x_[np.array([self.assign[ii]]).T.tolist()] += x[ii]
        ## mask
        x_ *= self.mask
        ## smooth
        x_ = self.domain.calc_smooth(x_,sigma=self.sigma)
        #return x_
        return field(self.domain,val=x_,target=kwargs.get("target",None))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _briefing(self,x,domain,inverse): ## > prepares x for `multiply`
        ## inspect x
        if(not isinstance(x,field)):
            x_ = field(domain,val=x,target=None)
        else:
            ## check x.domain
            if(domain==x.domain):
                x_ = x
            ## transform
            else:
                x_ = x.transform(target=domain,overwrite=False)
        ## weight if ...
        if(not self.imp)and(not domain.discrete)and(not inverse)and(self.den): ## respect density
            x_ = x_.weight(power=1,overwrite=False)
        return x_

    def _debriefing(self,x,x_,target,inverse): ## > evaluates x and x_ after `multiply`
        if(x_ is None):
            return None
        else:
            ## inspect x_
            if(not isinstance(x_,field)):
                x_ = field(target,val=x_,target=None)
            elif(x_.domain!=target):
                raise ValueError(about._errors.cstring("ERROR: invalid output domain."))
            ## weight if ...
            if(not self.imp)and(not target.discrete)and(not self.den^inverse): ## respect density
                x_ = x_.weight(power=-1,overwrite=False)
            ## inspect x
            if(isinstance(x,field)):
                ## repair ...
                if(self.domain==self.target!=x.domain):
                    x_ = x_.transform(target=x.domain,overwrite=False) ## ... domain
                if(x_.domain==x.domain)and(x_.target!=x.target):
                    x_.set_target(newtarget=x.target) ## ... codomain
            return x_

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty_core.response_operator>"

##-----------------------------------------------------------------------------



class invertible_operator(operator):
    """
        ..       __                                       __     __   __        __
        ..     /__/                                     /  /_  /__/ /  /      /  /
        ..     __   __ ___  __   __   _______   _____  /   _/  __  /  /___   /  /   _______
        ..   /  / /   _   ||  |/  / /   __  / /   __/ /  /   /  / /   _   | /  /  /   __  /
        ..  /  / /  / /  / |     / /  /____/ /  /    /  /_  /  / /  /_/  / /  /_ /  /____/
        .. /__/ /__/ /__/  |____/  \______/ /__/     \___/ /__/  \______/  \___/ \______/  operator class

        NIFTY subclass for invertible, self-adjoint (linear) operators

        The invertible operator class is an abstract class for self-adjoint or
        symmetric (linear) operators from which other more specific operator
        subclassescan be derived. Such operators inherit an automated inversion
        routine, namely conjugate gradient.

        Parameters
        ----------
        domain : space
            The space wherein valid arguments live.
        uni : bool, *optional*
            Indicates whether the operator is unitary or not.
            (default: False)
        imp : bool, *optional*
            Indicates whether the incorporation of volume weights in
            multiplications is already implemented in the `multiply`
            instance methods or not (default: False).
        para : {single object, tuple/list of objects}, *optional*
            This is a freeform tuple/list of parameters that derivatives of
            the operator class can use (default: None).

        See Also
        --------
        operator

        Notes
        -----
        This class is not meant to be instantiated. Operator classes derived
        from this one only need a `_multiply` or `_inverse_multiply` instance
        method to perform the other. However, one of them needs to be defined.

        Attributes
        ----------
        domain : space
            The space wherein valid arguments live.
        sym : bool
            Indicates whether the operator is self-adjoint or not.
        uni : bool
            Indicates whether the operator is unitary or not.
        imp : bool
            Indicates whether the incorporation of volume weights in
            multiplications is already implemented in the `multiply`
            instance methods or not.
        target : space
            The space wherein the operator output lives.
        para : {single object, list of objects}
            This is a freeform tuple/list of parameters that derivatives of
            the operator class can use. Not used in the base operators.

    """
    def __init__(self,domain,uni=False,imp=False,para=None):
        """
            Sets the standard operator properties.

            Parameters
            ----------
            domain : space
                The space wherein valid arguments live.
            uni : bool, *optional*
                Indicates whether the operator is unitary or not.
                (default: False)
            imp : bool, *optional*
                Indicates whether the incorporation of volume weights in
                multiplications is already implemented in the `multiply`
                instance methods or not (default: False).
            para : {single object, tuple/list of objects}, *optional*
                This is a freeform tuple/list of parameters that derivatives of
                the operator class can use (default: None).

        """
        if(not isinstance(domain,space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        self.domain = domain
        self.sym = True
        self.uni = bool(uni)

        if(self.domain.discrete):
            self.imp = True
        else:
            self.imp = bool(imp)

        self.target = self.domain

        if(para is not None):
            self.para = para

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _multiply(self,x,force=False,W=None,spam=None,reset=None,note=False,x0=None,tol=1E-4,clevel=1,limii=None,**kwargs):
        """
            Applies the invertible operator to a given field by invoking a
            conjugate gradient.

            Parameters
            ----------
            x : field
                Valid input field.
            force : bool
                Indicates wheter to return a field instead of ``None`` in case
                the conjugate gradient fails.

            Returns
            -------
            OIIx : field
                Mapped field with suitable domain.

            See Also
            --------
            conjugate_gradient

            Other Parameters
            ----------------
            W : {operator, function}, *optional*
                Operator `W` that is a preconditioner on `A` and is applicable to a
                field (default: None).
            spam : function, *optional*
                Callback function which is given the current `x` and iteration
                counter each iteration (default: None).
            reset : integer, *optional*
                Number of iterations after which to restart; i.e., forget previous
                conjugated directions (default: sqrt(b.dim())).
            note : bool, *optional*
                Indicates whether notes are printed or not (default: False).
            x0 : field, *optional*
                Starting guess for the minimization.
            tol : scalar, *optional*
                Tolerance specifying convergence; measured by current relative
                residual (default: 1E-4).
            clevel : integer, *optional*
                Number of times the tolerance should be undershot before
                exiting (default: 1).
            limii : integer, *optional*
                Maximum number of iterations performed (default: 10 * b.dim()).

        """
        x_,convergence = conjugate_gradient(self.inverse_times,x,W=W,spam=spam,reset=reset,note=note)(x0=x0,tol=tol,clevel=clevel,limii=limii)
        ## check convergence
        if(not convergence):
            if(not force)or(x_ is None):
                return None
            about.warnings.cprint("WARNING: conjugate gradient failed.")
        ## weight if ...
        if(not self.imp): ## continiuos domain/target
            x_.weight(power=-1,overwrite=True)
        return x_

    def _inverse_multiply(self,x,force=False,W=None,spam=None,reset=None,note=False,x0=None,tol=1E-4,clevel=1,limii=None,**kwargs):
        """
            Applies the inverse of the invertible operator to a given field by
            invoking a conjugate gradient.

            Parameters
            ----------
            x : field
                Valid input field.
            force : bool
                Indicates wheter to return a field instead of ``None`` in case
                the conjugate gradient fails.

            Returns
            -------
            OIx : field
                Mapped field with suitable domain.

            See Also
            --------
            conjugate_gradient

            Other Parameters
            ----------------
            W : {operator, function}, *optional*
                Operator `W` that is a preconditioner on `A` and is applicable to a
                field (default: None).
            spam : function, *optional*
                Callback function which is given the current `x` and iteration
                counter each iteration (default: None).
            reset : integer, *optional*
                Number of iterations after which to restart; i.e., forget previous
                conjugated directions (default: sqrt(b.dim())).
            note : bool, *optional*
                Indicates whether notes are printed or not (default: False).
            x0 : field, *optional*
                Starting guess for the minimization.
            tol : scalar, *optional*
                Tolerance specifying convergence; measured by current relative
                residual (default: 1E-4).
            clevel : integer, *optional*
                Number of times the tolerance should be undershot before
                exiting (default: 1).
            limii : integer, *optional*
                Maximum number of iterations performed (default: 10 * b.dim()).

        """
        x_,convergence = conjugate_gradient(self.times,x,W=W,spam=spam,reset=reset,note=note)(x0=x0,tol=tol,clevel=clevel,limii=limii)
        ## check convergence
        if(not convergence):
            if(not force)or(x_ is None):
                return None
            about.warnings.cprint("WARNING: conjugate gradient failed.")
        ## weight if ...
        if(not self.imp): ## continiuos domain/target
            x_.weight(power=1,overwrite=True)
        return x_

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty_tools.invertible_operator>"

##-----------------------------------------------------------------------------

##-----------------------------------------------------------------------------

class propagator_operator(operator):
    """
        ..                                                                            __
        ..                                                                          /  /_
        ..      _______   _____   ______    ______    ____ __   ____ __   ____ __  /   _/  ______    _____
        ..    /   _   / /   __/ /   _   | /   _   | /   _   / /   _   / /   _   / /  /   /   _   | /   __/
        ..   /  /_/  / /  /    /  /_/  / /  /_/  / /  /_/  / /  /_/  / /  /_/  / /  /_  /  /_/  / /  /
        ..  /   ____/ /__/     \______/ /   ____/  \______|  \___   /  \______|  \___/  \______/ /__/     operator class
        .. /__/                        /__/                 /______/

        NIFTY subclass for propagator operators (of a certain family)

        The propagator operators :math:`D` implemented here have an inverse
        formulation like :math:`(S^{-1} + M)`, :math:`(S^{-1} + N^{-1})`, or
        :math:`(S^{-1} + R^\dagger N^{-1} R)` as appearing in Wiener filter
        theory.

        Parameters
        ----------
        S : operator
            Covariance of the signal prior.
        M : operator
            Likelihood contribution.
        R : operator
            Response operator translating signal to (noiseless) data.
        N : operator
            Covariance of the noise prior or the likelihood, respectively.

        See Also
        --------
        conjugate_gradient

        Notes
        -----
        The propagator will puzzle the operators `S` and `M` or `R`, `N` or
        only `N` together in the predefined from, a domain is set
        automatically. The application of the inverse is done by invoking a
        conjugate gradient.
        Note that changes to `S`, `M`, `R` or `N` auto-update the propagator.

        Examples
        --------
        >>> f = field(rg_space(4), val=[2, 4, 6, 8])
        >>> S = power_operator(f.target, spec=1)
        >>> N = diagonal_operator(f.domain, diag=1)
        >>> D = propagator_operator(S=S, N=N) # D^{-1} = S^{-1} + N^{-1}
        >>> D(f).val
        array([ 1.,  2.,  3.,  4.])

        Attributes
        ----------
        domain : space
            A space wherein valid arguments live.
        codomain : space
            An alternative space wherein valid arguments live; commonly the
            codomain of the `domain` attribute.
        sym : bool
            Indicates that the operator is self-adjoint.
        uni : bool
            Indicates that the operator is not unitary.
        imp : bool
            Indicates that volume weights are implemented in the `multiply`
            instance methods.
        target : space
            The space wherein the operator output lives.
        _A1 : {operator, function}
            Application of :math:`S^{-1}` to a field.
        _A2 : {operator, function}
            Application of all operations not included in `A1` to a field.
        RN : {2-tuple of operators}, *optional*
            Contains `R` and `N` if given.

    """
    def __init__(self,S=None,M=None,R=None,N=None):
        """
            Sets the standard operator properties and `codomain`, `_A1`, `_A2`,
            and `RN` if required.

            Parameters
            ----------
            S : operator
                Covariance of the signal prior.
            M : operator
                Likelihood contribution.
            R : operator
                Response operator translating signal to (noiseless) data.
            N : operator
                Covariance of the noise prior or the likelihood, respectively.

        """
        ## check signal prior covariance
        if(S is None):
            raise Exception(about._errors.cstring("ERROR: insufficient input."))
        elif(not isinstance(S,operator)):
            raise ValueError(about._errors.cstring("ERROR: invalid input."))
        space1 = S.domain

        ## check likelihood (pseudo) covariance
        if(M is None):
            if(N is None):
                raise Exception(about._errors.cstring("ERROR: insufficient input."))
            elif(not isinstance(N,operator)):
                raise ValueError(about._errors.cstring("ERROR: invalid input."))
            if(R is None):
                space2 = N.domain
            elif(not isinstance(R,operator)):
                raise ValueError(about._errors.cstring("ERROR: invalid input."))
            else:
                space2 = R.domain
        elif(not isinstance(M,operator)):
            raise ValueError(about._errors.cstring("ERROR: invalid input."))
        else:
            space2 = M.domain

        ## set spaces
        self.domain = space2
        if(self.domain.check_codomain(space1)):
            self.codomain = space1
        else:
            raise ValueError(about._errors.cstring("ERROR: invalid input."))
        self.target = self.domain

        ## define A1 == S_inverse
        if(isinstance(S,diagonal_operator)):
            self._A1 = S._inverse_multiply ## S.imp == True
        else:
            self._A1 = S.inverse_times

        ## define A2 == M == R_adjoint N_inverse R == N_inverse
        if(M is None):
            if(R is not None):
                self.RN = (R,N)
                if(isinstance(N,diagonal_operator)):
                    self._A2 = self._standard_M_times_1
                else:
                    self._A2 = self._standard_M_times_2
            elif(isinstance(N,diagonal_operator)):
                self._A2 = N._inverse_multiply ## N.imp == True
            else:
                self._A2 = N.inverse_times
        elif(isinstance(M,diagonal_operator)):
            self._A2 = M._multiply ## M.imp == True
        else:
            self._A2 = M.times

        self.sym = True
        self.uni = False
        self.imp = True

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _standard_M_times_1(self,x,**kwargs): ## applies > R_adjoint N_inverse R assuming N is diagonal
        return self.RN[0].adjoint_times(self.RN[1]._inverse_multiply(self.RN[0].times(x))) ## N.imp = True

    def _standard_M_times_2(self,x,**kwargs): ## applies > R_adjoint N_inverse R
        return self.RN[0].adjoint_times(self.RN[1].inverse_times(self.RN[0].times(x)))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _inverse_multiply_1(self,x,**kwargs): ## > applies A1 + A2 in self.codomain
        return self._A1(x,pseudo=True)+self._A2(x.transform(self.domain)).transform(self.codomain)

    def _inverse_multiply_2(self,x,**kwargs): ## > applies A1 + A2 in self.domain
        return self._A1(x.transform(self.codomain),pseudo=True).transform(self.domain)+self._A2(x)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _briefing(self,x): ## > prepares x for `multiply`
        ## inspect x
        if(not isinstance(x,field)):
            return field(self.domain,val=x,target=None),False
        ## check x.domain
        elif(x.domain==self.domain):
            return x,False
        elif(x.domain==self.codomain):
            return x,True
        ## transform
        else:
            return x.transform(target=self.codomain,overwrite=False),True

    def _debriefing(self,x,x_,in_codomain): ## > evaluates x and x_ after `multiply`
        if(x_ is None):
            return None
        ## inspect x
        elif(isinstance(x,field)):
            ## repair ...
            if(in_codomain)and(x.domain!=self.codomain):
                    x_ = x_.transform(target=x.domain,overwrite=False) ## ... domain
            if(x_.target!=x.target):
                x_.set_target(newtarget=x.target) ## ... codomain
        return x_

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def times(self,x,force=False,W=None,spam=None,reset=None,note=False,x0=None,tol=1E-4,clevel=1,limii=None,**kwargs):
        """
            Applies the propagator to a given object by invoking a
            conjugate gradient.

            Parameters
            ----------
            x : {scalar, list, array, field}
                Scalars are interpreted as constant arrays, and an array will
                be interpreted as a field on the domain of the operator.
            force : bool
                Indicates wheter to return a field instead of ``None`` in case
                the conjugate gradient fails.

            Returns
            -------
            Dx : field
                Mapped field with suitable domain.

            See Also
            --------
            conjugate_gradient

            Other Parameters
            ----------------
            W : {operator, function}, *optional*
                Operator `W` that is a preconditioner on `A` and is applicable to a
                field (default: None).
            spam : function, *optional*
                Callback function which is given the current `x` and iteration
                counter each iteration (default: None).
            reset : integer, *optional*
                Number of iterations after which to restart; i.e., forget previous
                conjugated directions (default: sqrt(b.dim())).
            note : bool, *optional*
                Indicates whether notes are printed or not (default: False).
            x0 : field, *optional*
                Starting guess for the minimization.
            tol : scalar, *optional*
                Tolerance specifying convergence; measured by current relative
                residual (default: 1E-4).
            clevel : integer, *optional*
                Number of times the tolerance should be undershot before
                exiting (default: 1).
            limii : integer, *optional*
                Maximum number of iterations performed (default: 10 * b.dim()).

        """
        ## prepare
        x_,in_codomain = self._briefing(x)
        ## apply operator
        if(in_codomain):
            A = self._inverse_multiply_1
        else:
            A = self._inverse_multiply_2
        x_,convergence = conjugate_gradient(A,x_,W=W,spam=spam,reset=reset,note=note)(x0=x0,tol=tol,clevel=clevel,limii=limii)
        ## evaluate
        if(not convergence):
            if(not force):
                return None
            about.warnings.cprint("WARNING: conjugate gradient failed.")
        return self._debriefing(x,x_,in_codomain)

    def inverse_times(self,x,**kwargs):
        """
            Applies the inverse propagator to a given object.

            Parameters
            ----------
            x : {scalar, list, array, field}
                Scalars are interpreted as constant arrays, and an array will
                be interpreted as a field on the domain of the operator.

            Returns
            -------
            DIx : field
                Mapped field with suitable domain.

        """
        ## prepare
        x_,in_codomain = self._briefing(x)
        ## apply operator
        if(in_codomain):
            x_ = self._inverse_multiply_1(x_)
        else:
            x_ = self._inverse_multiply_2(x_)
        ## evaluate
        return self._debriefing(x,x_,in_codomain)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty_tools.propagator_operator>"

##-----------------------------------------------------------------------------
