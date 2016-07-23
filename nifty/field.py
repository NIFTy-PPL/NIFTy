from __future__ import division
import numpy as np
import pylab as pl

from d2o import distributed_data_object, \
    STRATEGIES as DISTRIBUTION_STRATEGIES

from nifty.config import about, \
    nifty_configuration as gc, \
    dependency_injector as gdi

from nifty.field_types import FieldType,\
                              FieldArray

from nifty.spaces.space import Space

import nifty.nifty_utilities as utilities
from nifty_random import random

POINT_DISTRIBUTION_STRATEGIES = DISTRIBUTION_STRATEGIES['global']


class Field(object):
    """
        ..         ____   __             __          __
        ..       /   _/ /__/           /  /        /  /
        ..      /  /_   __   _______  /  /    ____/  /
        ..     /   _/ /  / /   __  / /  /   /   _   /
        ..    /  /   /  / /  /____/ /  /_  /  /_/  /
        ..   /__/   /__/  \______/  \___/  \______|  class

        Basic NIFTy class for fields.

        Parameters
        ----------
        domain : space
            The space wherein valid arguments live.

        val : {scalar, ndarray}, *optional*
            Defines field values, either to be given by a number interpreted
            as a constant array, or as an arbitrary array consistent with the
            space defined in domain or to be drawn from a random distribution
            controlled by kwargs.

        codomain : space, *optional*
            The space wherein the operator output lives (default: domain).


        Other Parameters
        ----------------
        random : string
            Indicates that the field values should be drawn from a certain
            distribution using a pseudo-random number generator.
            Supported distributions are:

            - "pm1" (uniform distribution over {+1,-1} or {+1,+i,-1,-i}
            - "gau" (normal distribution with zero-mean and a given standard
                deviation or variance)
            - "syn" (synthesizes from a given power spectrum)
            - "uni" (uniform distribution over [vmin,vmax[)

        dev : scalar
            Sets the standard deviation of the Gaussian distribution
            (default=1).

        var : scalar
            Sets the variance of the Gaussian distribution, outranking the dev
            parameter (default=1).

        spec : {scalar, list, array, field, function}
            Specifies a power spectrum from which the field values should be
            synthesized (default=1). Can be given as a constant, or as an
            array with indvidual entries per mode.
        log : bool
            Flag specifying if the spectral binning is performed on logarithmic
            scale or not; if set, the number of used bins is set
            automatically (if not given otherwise); by default no binning
            is done (default: None).
        nbin : integer
            Number of used spectral bins; if given `log` is set to ``False``;
            integers below the minimum of 3 induce an automatic setting;
            by default no binning is done (default: None).
        binbounds : {list, array}
            User specific inner boundaries of the bins, which are preferred
            over the above parameters; by default no binning is done
            (default: None).

        vmin : scalar
            Sets the lower limit for the uniform distribution.
        vmax : scalar
            Sets the upper limit for the uniform distribution.

        Attributes
        ----------
        domain : space
            The space wherein valid arguments live.

        val : {scalar, ndarray}, *optional*
            Defines field values, either to be given by a number interpreted
            as a constant array, or as an arbitrary array consistent with the
            space defined in domain or to be drawn from a random distribution
            controlled by the keyword arguments.

        codomain : space, *optional*
            The space wherein the operator output lives (default: domain).

    """

    def __init__(self, domain=None, val=None, codomain=None,
                 dtype=None, field_type=None, copy=False,
                 datamodel=None, **kwargs):
        """
            Sets the attributes for a field class instance.

        Parameters
        ----------
        domain : space
            The space wherein valid arguments live.

        val : {scalar,ndarray}, *optional*
            Defines field values, either to be given by a number interpreted
            as a constant array, or as an arbitrary array consistent with the
            space defined in domain or to be drawn from a random distribution
            controlled by the keyword arguments.

        codomain : space, *optional*
            The space wherein the operator output lives (default: domain).

        Returns
        -------
        Nothing

        """
        # If the given val was a field, try to cast it accordingly to the given
        # domain and codomain, etc...
        if isinstance(val, Field):
            self._init_from_field(f=val,
                                  domain=domain,
                                  codomain=codomain,
                                  copy=copy,
                                  dtype=dtype,
                                  field_type=field_type,
                                  datamodel=datamodel,
                                  **kwargs)
        else:
            self._init_from_array(val=val,
                                  domain=domain,
                                  codomain=codomain,
                                  copy=copy,
                                  dtype=dtype,
                                  field_type=field_type,
                                  datamodel=datamodel,
                                  **kwargs)

    def _init_from_field(self, f, domain, codomain, copy, dtype,
                         field_type, datamodel, **kwargs):
        # check domain
        if domain is None:
            domain = f.domain

        # check codomain
        if codomain is None:
            if self._check_codomain(domain, f.codomain):
                codomain = f.codomain
            else:
                codomain = self.get_codomain(domain)

        # Check if the given field lives in a space which is compatible to the
        # given domain
        if f.domain != domain:
            # Try to transform the given field to the given domain/codomain
            f = f.transform(new_domain=domain,
                            new_codomain=codomain)

        self._init_from_array(domain=domain,
                              val=f.val,
                              codomain=codomain,
                              copy=copy,
                              dtype=dtype,
                              datamodel=datamodel,
                              **kwargs)

    def _init_from_array(self, val, domain, codomain, copy, dtype,
                         field_type, datamodel, **kwargs):
        # check domain
        self.domain = self._parse_domain(domain=domain)
        self.domain_axes = self._get_axes_tuple(self.domain)

        # check codomain
        if codomain is None:
            self.codomain = self._build_codomain(domain=self.domain)
        else:
            self.codomain = self._parse_codomain(codomain, self.domain)

        self.field_type = self._parse_field_type(field_type)
        self.field_type_axes = self._get_axes_tuple(self.field_type)

        if dtype is None:
            dtype = self._infer_dtype(domain=self.domain,
                                      dtype=dtype,
                                      field_type=self.field_type)
        self.dtype = dtype

        self._comm = getattr(gdi[gc['mpi_module']], gc['default_comm'])

        if datamodel in DISTRIBUTION_STRATEGIES['all']:
            self.datamodel = datamodel
        elif isinstance(val, distributed_data_object):
            self.datamodel = val.distribution_strategy
        else:
            self.datamodel = gc['default_datamodel']

        if val is None:
            if kwargs == {}:
                val = self.cast(0)
            else:
                val = self.get_random_values(domain=self.domain,
                                             codomain=self.codomain,
                                             **kwargs)
        self.set_val(new_val=val, copy=copy)

    def _infer_dtype(self, domain=None, dtype=None, field_type=None):
        if dtype is None:
            dtype_tuple = (np.dtype(gc['default_field_dtype']),)
        else:
            dtype_tuple = (np.dtype(dtype))
        if domain is not None:
            dtype_tuple += tuple(np.dtype(sp.dtype) for sp in domain)
        if field_type is not None:
            dtype_tuple += tuple(np.dtype(ft.dtype) for ft in field_type)

        dtype = reduce(lambda x, y: np.result_type(x, y), dtype_tuple)
        return dtype

    def _get_axes_tuple(self, things_with_shape):
        i = 0
        axes_list = []
        for thing in things_with_shape:
            l = []
            for j in range(len(thing.shape)):
                l += [i]
                i += 1
            axes_list += [tuple(l)]
        return tuple(axes_list)

    def _parse_domain(self, domain):
        if domain is None:
            domain = ()
        elif not isinstance(domain, tuple):
            domain = (domain,)
        for d in domain:
            if not isinstance(d, Space):
                raise TypeError(about._errors.cstring(
                    "ERROR: Given domain contains something that is not a "
                    "nifty.space."))
        return domain

    def _parse_codomain(self, codomain, domain):
        if not isinstance(codomain, tuple):
            codomain = (codomain,)
        if len(domain) != len(codomain):
            raise ValueError(about._errors.cstring(
                "ERROR: domain and codomain do not have the same length."))
        for (cd, d) in zip(codomain, domain):
            if not isinstance(cd, Space):
                raise TypeError(about._errors.cstring(
                    "ERROR: Given codomain contains something that is not a"
                    "nifty.space."))
            if not d.check_codomain(cd):
                raise ValueError(about._errors.cstring(
                    "ERROR: codomain contains a space that is not compatible "
                    "to its domain-counterpart."))
        return codomain

    def _parse_field_type(self, field_type):
        if field_type is None:
            field_type = ()
        elif not isinstance(field_type, tuple):
            field_type = (field_type,)
        for ft in field_type:
            if not isinstance(ft, FieldType):
                raise TypeError(about._errors.cstring(
                    "ERROR: Given object is not a nifty.FieldType."))
        return field_type

    def _build_codomain(self, domain):
        codomain = tuple(sp.get_codomain() for sp in domain)
        return codomain

    def get_random_values(self, **kwargs):
        arg = random.parse_arguments(self, **kwargs)

        if arg is None:
            return self.cast(0)

        # Prepare the empty distributed_data_object
        sample = distributed_data_object(
                                    global_shape=self.shape,
                                    dtype=self.dtype)

        # Case 1: uniform distribution over {-1,+1}/{1,i,-1,-i}
        if arg['random'] == 'pm1':
            sample.apply_generator(lambda s: random.pm1(dtype=self.dtype,
                                                        shape=s))

        # Case 2: normal distribution with zero-mean and a given standard
        #         deviation or variance
        elif arg['random'] == 'gau':
            std = arg['std']
            if np.isscalar(std) or std is None:
                processed_std = std
            else:
                try:
                    processed_std = sample.distributor. \
                        extract_local_data(std)
                except(AttributeError):
                    processed_std = std

            sample.apply_generator(lambda s: random.gau(dtype=self.dtype,
                                                        shape=s,
                                                        mean=arg['mean'],
                                                        std=processed_std))

        # Case 3: uniform distribution
        elif arg['random'] == 'uni':
            sample.apply_generator(lambda s: random.uni(dtype=self.dtype,
                                                        shape=s,
                                                        vmin=arg['vmin'],
                                                        vmax=arg['vmax']))
        return sample

    def __len__(self):
        return int(self.dim[0])

    def copy(self, domain=None, codomain=None, field_type=None, **kwargs):
        copied_val = self._unary_operation(self.get_val(), op='copy', **kwargs)
        # TODO: respect distribution_strategy
        new_field = self.copy_empty(domain=domain,
                                    codomain=codomain,
                                    field_type=field_type)
        new_field.set_val(new_val=copied_val, copy=True)
        return new_field

    def _fast_copy_empty(self):
        # make an empty field
        new_field = EmptyField()
        # repair its class
        new_field.__class__ = self.__class__
        # copy domain, codomain and val
        for key, value in self.__dict__.items():
            if key != 'val':
                new_field.__dict__[key] = value
            else:
                new_field.__dict__[key] = \
                    self._unary_operation(self.val, op='copy_empty')
        return new_field

    def copy_empty(self, domain=None, codomain=None, dtype=None,
                   datamodel=None, field_type=None, **kwargs):
        if domain is None:
            domain = self.domain

        if codomain is None:
            codomain = self.codomain

        if dtype is None:
            dtype = self.dtype

        if datamodel is None:
            datamodel = self.datamodel

        if field_type is None:
            field_type = self.field_type

        _fast_copyable = True
        for i in xrange(len(self.domain)):
            if self.domain[i] is not domain[i]:
                _fast_copyable = False
                break
            if self.codomain[i] is not codomain[i]:
                _fast_copyable = False
                break

        for i in xrange(len(self.field_type)):
            if self.field_type[i] is not field_type[i]:
                _fast_copyable = False
                break

        if (_fast_copyable and dtype == self.dtype and
                datamodel == self.datamodel and kwargs == {}):
            new_field = self._fast_copy_empty()
        else:
            new_field = Field(domain=domain, codomain=codomain, dtype=dtype,
                              datamodel=datamodel, field_type=field_type,
                              **kwargs)
        return new_field

    # TODO: use property for val
    def set_val(self, new_val=None, copy=False):
        """
            Resets the field values.

            Parameters
            ----------
            new_val : {scalar, ndarray}
                New field values either as a constant or an arbitrary array.

        """
        new_val = self.cast(new_val)
        if copy:
            new_val = self.unary_operation(new_val, op='copy')
        self.val = new_val
        return self.val

    def get_val(self, copy=False):
        if copy:
            return self.val.copy()
        else:
            return self.val

    def __getitem__(self, key):
        return self.val[key]

    def __setitem__(self, key, item):
        self.val[key] = item

    @property
    def shape(self):
        shape_tuple = ()
        shape_tuple += tuple(sp.shape for sp in self.domain)
        shape_tuple += tuple(ft.shape for ft in self.field_type)
        try:
            global_shape = reduce(lambda x, y: x + y, shape_tuple)
        except TypeError:
            global_shape = ()

        return global_shape

    # use space.dim and field_type.dim
    @property
    def dim(self):
        """
            Computes the (array) dimension of the underlying space.

            Parameters
            ----------
            split : bool
                Sets the output to be either split up per axis or
                in form of total number of field entries in all
                dimensions (default=False)

            Returns
            -------
            dim : {scalar, ndarray}
                Dimension of space.

        """
        return reduce(lambda x, y: x * y, self.shape)

    @property
    def dof(self):
        dof_tuple = ()
        dof_tuple += tuple(sp.dof for sp in self.domain)
        dof_tuple += tuple(ft.dof for ft in self.field_type)
        try:
            return reduce(lambda x, y: x * y, dof_tuple)
        except TypeError:
            return ()

    def cast(self, x=None, dtype=None):
        if dtype is None:
            dtype = self.dtype
        else:
            dtype = np.dtype(dtype)

        casted_x = self._cast_to_d2o(x, dtype=dtype)

        for ind, sp in enumerate(self.domain):
            casted_x = sp.complement_cast(casted_x,
                                          axis=self.domain_axes[ind])

        for ind, ft in enumerate(self.field_type):
            casted_x = ft.complement_cast(casted_x,
                                          axis=self.field_type_axes[ind])

        return casted_x

    def _cast_to_d2o(self, x, dtype=None, shape=None, **kwargs):
        """
            Computes valid field values from a given object, trying
            to translate the given data into a valid form. Thereby it is as
            benevolent as possible.

            Parameters
            ----------
            x : {float, numpy.ndarray, nifty.field}
                Object to be transformed into an array of valid field values.

            Returns
            -------
            x : numpy.ndarray, distributed_data_object
                Array containing the field values, which are compatible to the
                space.

            Other parameters
            ----------------
            verbose : bool, *optional*
                Whether the method should raise a warning if information is
                lost during casting (default: False).
        """
        if isinstance(x, Field):
            x = x.get_val()

        if dtype is None:
            dtype = self.dtype

        if shape is None:
            shape = self.shape

        # Case 1: x is a distributed_data_object
        # TODO: Use d2o casting for this case directly, too.
        if isinstance(x, distributed_data_object):
            if x.comm is not self._comm:
                raise ValueError(about._errors.cstring(
                    "ERROR: comms do not match."))
            to_copy = False

            # Check the shape
            if np.any(np.array(x.shape) != np.array(shape)):
                # Check if at least the number of degrees of freedom is equal
                if x.dim == self.dim:
                    try:
                        temp = x.copy_empty(global_shape=shape)
                        temp.set_local_data(x, copy=False)
                    except:
                        # If the number of dof is equal or 1, use np.reshape...
                        about.warnings.cflush(
                            "WARNING: Trying to reshape the data. This " +
                            "operation is expensive as it consolidates the " +
                            "full data!\n")
                        temp = x
                        temp = np.reshape(temp, shape)
                    # ... and cast again
                    return self._cast_to_d2o(temp, dtype=dtype, **kwargs)

                else:
                    raise ValueError(about._errors.cstring(
                        "ERROR: Data has incompatible shape!"))

            # Check the dtype
            if x.dtype != dtype:
                if x.dtype > dtype:
                    about.warnings.cflush(
                        "WARNING: Datatypes are of conflicting precision " +
                        "(own: " + str(dtype) + " <> foreign: " +
                        str(x.dtype) + ") and will be casted! Potential " +
                        "loss of precision!\n")
                to_copy = True

            # Check the distribution_strategy
            if x.distribution_strategy != self.datamodel:
                to_copy = True

            if to_copy:
                temp = x.copy_empty(dtype=dtype,
                                    distribution_strategy=self.datamodel)
                temp.set_data(to_key=(slice(None),),
                              data=x,
                              from_key=(slice(None),))
                temp.hermitian = x.hermitian
                x = temp

            return x

        # Case 2: x is something else
        # Use general d2o casting
        else:
            x = distributed_data_object(x,
                                        global_shape=self.shape,
                                        dtype=dtype,
                                        distribution_strategy=self.datamodel,
                                        comm=self._comm)
            # Cast the d2o
            return self.cast(x, dtype=dtype)

    def weight(self, power=1, inplace=False, spaces=None):
        """
            Returns the field values, weighted with the volume factors to a
            given power. The field values will optionally be overwritten.

            Parameters
            ----------
            power : scalar, *optional*
                Specifies the optional power coefficient to which the field
                values are taken (default=1).

            inplace : bool, *optional*
                Whether to overwrite the field values or not (default: False).

            Returns
            -------
            field   : field, *optional*
                If inplace is False, the weighted field is returned.
                Otherwise, nothing is returned.

        """
        if inplace:
            new_field = self
        else:
            new_field = self.copy_empty()

        new_val = self.get_val(copy=False)

        if spaces is None:
            spaces = range(len(self.shape))

        for ind, sp in enumerate(self.domain):
            new_val = sp.weight(new_val,
                                power=power,
                                axes=self.domain_axes[ind],
                                inplace=inplace)

        new_field.set_val(new_val=new_val, copy=False)
        return new_field

    def norm(self, q=2):
        """
            Computes the Lq-norm of the field values.

            Parameters
            ----------
            q : scalar
                Parameter q of the Lq-norm (default: 2).

            Returns
            -------
            norm : scalar
                The Lq-norm of the field values.

        """
        if q == 2:
            return (self.dot(x=self)) ** (1 / 2)
        else:
            return self.dot(x=self ** (q - 1)) ** (1 / q)

    def dot(self, x=None, bare=False):
        """
            Computes the inner product of the field with a given object
            implying the correct volume factor needed to reflect the
            discretization of the continuous fields.

            Parameters
            ----------
            x : {scalar, ndarray, field}, *optional*
                The object with which the inner product is computed
                (default=None).

            Returns
            -------
            dot : scalar
                The result of the inner product.

        """
        # Case 1: x equals None
        if x is None:
            return None

        # Case 2: x is a field
        elif isinstance(x, Field):
            for ind, sp in enumerate(self.domain):
                assert sp == x.domain[ind]

            # whether the domain matches exactly or not:
            # extract the data from x and try to dot with this
            return self.dot(x=x.get_val(), bare=bare)

        # Case 3: x is something else
        else:

            # Compute the dot respecting the fact of discrete/continous spaces
            if not bare:
                y = self.weight(power=1)
            else:
                y = self
            y = y.get_val(copy=False)

            # Cast the input in order to cure dtype and shape differences
            x = self.cast(x)

            dotted = x.conjugate() * y

            for ind in range(-1, -len(self.field_type_axes)-1, -1):
                dotted = self.field_type[ind].dot_contraction(
                            dotted,
                            axes=self.field_type_axes[ind])

            for ind in range(-1, -len(self.domain_axes)-1, -1):
                dotted = self.domain[ind].dot_contraction(
                            dotted,
                            axes=self.domain_axes[ind])
            return dotted

    def vdot(self, *args, **kwargs):
        return self.dot(*args, **kwargs)

#    def outer_dot(self, x=1, axis=None):
#
#        # Use the fact that self.val is a numpy array of dtype np.object
#        # -> The shape casting, etc... can be done by numpy
#        # If ishape == (), self.val will be multiplied with x directly.
#        if self.ishape == ():
#            return self * x
#        new_val = np.sum(self.get_val() * x, axis=axis)
#        # if axis != None, the contraction was not overarching
#        if np.dtype(new_val.dtype).type == np.object_:
#            new_field = self.copy_empty(ishape=new_val.shape)
#        else:
#            new_field = self.copy_empty(ishape=())
#        new_field.set_val(new_val=new_val)
#        return new_field
#
#    def tensor_product(self, x=None):
#        if x is None:
#            return self
#        elif np.isscalar(x) == True:
#            return self * x
#        else:
#            if self.ishape == ():
#                temp_val = self.get_val()
#                old_val = np.empty((1,), dtype=np.object)
#                old_val[0] = temp_val
#            else:
#                old_val = self.get_val()
#
#            new_val = np.tensordot(old_val, x, axes=0)
#
#            if self.ishape == ():
#                new_val = new_val[0]
#            new_field = self.copy_empty(ishape=new_val.shape)
#            new_field.set_val(new_val=new_val)
#
#            return new_field

    def conjugate(self, inplace=False):
        """
            Computes the complex conjugate of the field.

            Returns
            -------
            cc : field
                The complex conjugated field.

        """
        if inplace:
            work_field = self
        else:
            work_field = self.copy_empty()

        new_val = self.get_val(copy=False)
        new_val = self._unary_operation(new_val, op='conjugate')

        work_field.set_val(new_val=new_val, copy=False)

        return work_field

    def transform(self, spaces=None, **kwargs):
        """
            Computes the transform of the field using the appropriate conjugate
            transformation.

            Parameters
            ----------
            codomain : space, *optional*
                Domain of the transform of the field (default:self.codomain)

            overwrite : bool, *optional*
                Whether to overwrite the field or not (default: False).

            Other Parameters
            ----------------
            iter : scalar
                Number of iterations (default: 0)

            Returns
            -------
            field : field, *optional*
                If overwrite is False, the transformed field is returned.
                Otherwise, nothing is returned.

        """

        try:
            iter(spaces)
        except TypeError:
            if spaces is None:
                spaces = xrange(len(self.domain_axes))
            else:
                spaces = (spaces, )

        new_val = self.get_val()
        new_domain = ()
        new_codomain = ()
        for ind in xrange(len(self.domain)):
            if ind in spaces:
                sp = self.domain[ind]
                cosp = self.codomain[ind]
                new_val = sp.calc_transform(new_val,
                                            codomain=cosp,
                                            axes=self.domain_axes[ind],
                                            **kwargs)
                new_domain += (self.codomain[ind],)
                new_codomain += (self.domain[ind],)
            else:
                new_domain += (self.domain[ind],)
                new_codomain += (self.codomain[ind],)

        return_field = self.copy_empty(domain=new_domain,
                                       codomain=new_codomain)
        return_field.set_val(new_val=new_val, copy=False)

        return return_field

    def smooth(self, sigma=0, spaces=None, **kwargs):
        """
            Smoothes the field by convolution with a Gaussian kernel.

            Parameters
            ----------
            sigma : scalar, *optional*
                standard deviation of the Gaussian kernel specified in units of
                length in position space (default: 0)

            overwrite : bool, *optional*
                Whether to overwrite the field or not (default: False).

            Other Parameters
            ----------------
            iter : scalar
                Number of iterations (default: 0)

            Returns
            -------
            field : field, *optional*
                If overwrite is False, the transformed field is returned.
                Otherwise, nothing is returned.

        """
        new_field = self.copy_empty()

        try:
            spaces_iterator = iter(spaces)
        except TypeError:
            if spaces is None:
                spaces_iterator = xrange(len(self.domain))
            else:
                spaces_iterator = (spaces, )

        new_val = self.get_val()
        for ind in spaces_iterator:
            sp = self.domain[ind]
            new_val = sp.calc_smooth(new_val,
                                     sigma=sigma,
                                     axes=self.domain_axes[ind],
                                     **kwargs)

        new_field.set_val(new_val=new_val, copy=False)
        return new_field

    def power(self, **kwargs):
        """
            Computes the power spectrum of the field values.

            Other Parameters
            ----------------
            pindex : ndarray, *optional*
                Specifies the indexing array for the distribution of
                indices in conjugate space (default: None).
            kindex : numpy.ndarray, *optional*
                Scale corresponding to each band in the power spectrum
                (default: None).
            rho : scalar
                Number of degrees of freedom per irreducible band
                (default=None).
            log : bool, *optional*
                Flag specifying if the spectral binning is performed on
                logarithmic
                scale or not; if set, the number of used bins is set
                automatically (if not given otherwise); by default no binning
                is done (default: None).
            nbin : integer, *optional*
                Number of used spectral bins; if given `log` is set to
                ``False``;
                integers below the minimum of 3 induce an automatic setting;
                by default no binning is done (default: None).
            binbounds : {list, array}, *optional*
                User specific inner boundaries of the bins, which are preferred
                over the above parameters; by default no binning is done
                (default: None).
            iter : scalar
                Number of iterations (default: 0)

            Returns
            -------
            spec : ndarray
                Returns the power spectrum.

        """
        if ("codomain" in kwargs):
            kwargs.__delitem__("codomain")
            about.warnings.cprint("WARNING: codomain was removed from kwargs.")

#        power_spectrum = self.get_val()
#        for ind, space in self.domain:
#            power_spectrum = space.calc_smooth(power_spectrum,
#                                               codomain=self.codomain,
#                                               axis=self.axes_list[ind],
#                                               **kwargs)
#
#        return power_spectrum

    def hat(self):
        """
            Translates the field into a diagonal operator.

            Returns
            -------
            D : operator
                The new diagonal operator instance.

        """
        from nifty.operators.nifty_operators import diagonal_operator
        return diagonal_operator(domain=self.domain,
                                 diag=self.get_val(),
                                 bare=False,
                                 ishape=self.ishape)

    def inverse_hat(self):
        """
            Translates the inverted field into a diagonal operator.

            Returns
            -------
            D : operator
                The new diagonal operator instance.

        """
        any_zero_Q = np.any(map(lambda z: (z == 0), self.get_val()))
        if any_zero_Q:
            raise AttributeError(
                about._errors.cstring("ERROR: singular operator."))
        else:
            from nifty.operators.nifty_operators import diagonal_operator
            return diagonal_operator(domain=self.domain,
                                     diag=(1 / self).get_val(),
                                     bare=False,
                                     ishape=self.ishape)

    def plot(self, **kwargs):
        """
            Plots the field values using matplotlib routines.

            Other Parameters
            ----------------
            title : string
                Title of the plot (default= "").
            vmin : scalar
                Minimum value displayed (default=min(x)).
            vmax : scalar
                Maximum value displayed (default=max(x)).
            power : bool
                Whether to plot the power spectrum or the array (default=None).
            unit : string
                The unit of the field values (default="").
            norm : scalar
                A normalization (default=None).
            cmap : cmap
                A color map (default=None).
            cbar : bool
                Whether to show the color bar or not (default=True).
            other : {scalar, ndarray, field}
                Object or tuple of objects to be added (default=None).
            legend : bool
                Whether to show the legend or not (default=False).
            mono : bool
                Whether to plot the monopol of the power spectrum or not
                (default=True).
            save : string, *optional*
                Valid file name where the figure is to be stored, by default
                the figure is not saved (default: False).
            error : {scalar, ndarray, field}
                object indicating some confidence intervall (default=None).
            iter : scalar
                Number of iterations (default: 0).
            kindex : scalar
                The spectral index per irreducible band (default=None).
            log : bool, *optional*
                Flag specifying if the spectral binning is performed on
                logarithmic
                scale or not; if set, the number of used bins is set
                automatically (if not given otherwise); by default no binning
                is done (default: None).
            nbin : integer, *optional*
                Number of used spectral bins; if given `log` is set to
                ``False``;
                integers below the minimum of 3 induce an automatic setting;
                by default no binning is done (default: None).
            binbounds : {list, array}, *optional*
                User specific inner boundaries of the bins, which are preferred
                over the above parameters; by default no binning is done
                (default: None).

            Notes
            -----
            The applicability of the keyword arguments depends on the
            respective space on which the field is defined. Confer to the
            corresponding :py:meth:`get_plot` method.

        """
        # if a save path is given, set pylab to not-interactive
        remember_interactive = pl.isinteractive()
        pl.matplotlib.interactive(not bool(kwargs.get("save", False)))

        if "codomain" in kwargs:
            kwargs.__delitem__("codomain")
            about.warnings.cprint("WARNING: codomain was removed from kwargs.")

        # draw/save the plot(s)
        self.domain.get_plot(self.val, codomain=self.codomain, **kwargs)

        # restore the pylab interactiveness
        pl.matplotlib.interactive(remember_interactive)

    def __repr__(self):
        return "<nifty_core.field>"

    def __str__(self):
        minmax = [self.min(), self.max()]
        mean = self.mean()
        return "nifty_core.field instance\n- domain      = " + \
               repr(self.domain) + \
               "\n- val         = " + repr(self.get_val()) + \
               "\n  - min.,max. = " + str(minmax) + \
               "\n  - mean = " + str(mean) + \
               "\n- codomain      = " + repr(self.codomain) + \
               "\n- ishape          = " + str(self.ishape)

    def sum(self, **kwargs):
        return self._unary_operation(self.get_val(), op='sum', **kwargs)

    def prod(self, **kwargs):
        return self._unary_operation(self.get_val(), op='prod', **kwargs)

    def all(self, **kwargs):
        return self._unary_operation(self.get_val(), op='all', **kwargs)

    def any(self, **kwargs):
        return self._unary_operation(self.get_val(), op='any', **kwargs)

    def min(self, ignore=False, **kwargs):
        """
            Returns the minimum of the field values.

            Parameters
            ----------
            ignore : bool
                Whether to ignore NANs or not (default: False).

            Returns
            -------
            amin : {scalar, ndarray}
                Minimum field value.

            See Also
            --------
            np.amin, np.nanmin

        """
        return self._unary_operation(self.get_val(), op='amin', **kwargs)

    def nanmin(self, **kwargs):
        return self._unary_operation(self.get_val(), op='nanmin', **kwargs)

    def max(self, **kwargs):
        """
            Returns the maximum of the field values.

            Parameters
            ----------
            ignore : bool
                Whether to ignore NANs or not (default: False).

            Returns
            -------
            amax : {scalar, ndarray}
                Maximum field value.

            See Also
            --------
            np.amax, np.nanmax

        """
        return self._unary_operation(self.get_val(), op='amax', **kwargs)

    def nanmax(self, **kwargs):
        return self._unary_operation(self.get_val(), op='nanmax', **kwargs)

    def median(self, **kwargs):
        """
            Returns the median of the field values.

            Returns
            -------
            med : scalar
                Median field value.

            See Also
            --------
            np.median

        """
        return self._unary_operation(self.get_val(), op='median',
                                     **kwargs)

    def mean(self, **kwargs):
        """
            Returns the mean of the field values.

            Returns
            -------
            mean : scalar
                Mean field value.

            See Also
            --------
            np.mean

        """
        return self._unary_operation(self.get_val(), op='mean',
                                     **kwargs)

    def std(self, **kwargs):
        """
            Returns the standard deviation of the field values.

            Returns
            -------
            std : scalar
                Standard deviation of the field values.

            See Also
            --------
            np.std

        """
        return self._unary_operation(self.get_val(), op='std',
                                     **kwargs)

    def var(self, **kwargs):
        """
            Returns the variance of the field values.

            Returns
            -------
            var : scalar
                Variance of the field values.

            See Also
            --------
            np.var

        """
        return self._unary_operation(self.get_val(), op='var',
                                     **kwargs)

    # TODO: replace `split` by `def argmin_nonflat`
    def argmin(self, split=False, **kwargs):
        """
            Returns the index of the minimum field value.

            Parameters
            ----------
            split : bool
                Whether to split (unravel) the flat index or not; does not
                apply to multiple indices along some axis (default: True).

            Returns
            -------
            ind : {integer, tuple, array}
                Index of the minimum field value being an integer for
                one-dimensional fields, a tuple for multi-dimensional fields,
                and an array in case minima along some axis are requested.

            See Also
            --------
            np.argmax, np.argmin

        """
        if split:
            return self._unary_operation(self.get_val(), op='argmin_nonflat',
                                         **kwargs)
        else:
            return self._unary_operation(self.get_val(), op='argmin',
                                         **kwargs)

    def argmax(self, split=False, **kwargs):
        """
            Returns the index of the maximum field value.

            Parameters
            ----------
            split : bool
                Whether to split (unravel) the flat index or not; does not
                apply to multiple indices along some axis (default: True).

            Returns
            -------
            ind : {integer, tuple, array}
                Index of the maximum field value being an integer for
                one-dimensional fields, a tuple for multi-dimensional fields,
                and an array in case maxima along some axis are requested.

            See Also
            --------
            np.argmax, np.argmin

        """
        if split:
            return self._unary_operation(self.get_val(), op='argmax_nonflat',
                                         **kwargs)
        else:
            return self._unary_operation(self.get_val(), op='argmax',
                                         **kwargs)

    # TODO: Implement the full range of unary and binary operotions

    def __pos__(self):
        new_field = self.copy_empty()
        new_val = self._unary_operation(self.get_val(), op='pos')
        new_field.set_val(new_val=new_val)
        return new_field

    def __neg__(self):
        new_field = self.copy_empty()
        new_val = self._unary_operation(self.get_val(), op='neg')
        new_field.set_val(new_val=new_val)
        return new_field

    def __abs__(self):
        new_field = self.copy_empty()
        new_val = self._unary_operation(self.get_val(), op='abs')
        new_field.set_val(new_val=new_val)
        return new_field

    def _binary_helper(self, other, op='None', inplace=False):
        # if other is a field, make sure that the domains match
        if isinstance(other, Field):
            other = Field(domain=self.domain,
                          val=other,
                          codomain=self.codomain,
                          copy=False)
        try:
            other_val = other.get_val()
        except AttributeError:
            other_val = other

        # bring other_val into the right shape
        other_val = self._cast_to_d2o(other_val)

        new_val = map(
            lambda z1, z2: self._binary_operation(z1, z2, op=op, cast=0),
            self.get_val(),
            other_val)

        if inplace:
            working_field = self
        else:
            working_field = self.copy_empty()

        working_field.set_val(new_val=new_val)
        return working_field

    def _unary_operation(self, x, op='None', axis=None, **kwargs):
        """
        x must be a numpy array which is compatible with the space!
        Valid operations are

        """
        translation = {'pos': lambda y: getattr(y, '__pos__')(),
                       'neg': lambda y: getattr(y, '__neg__')(),
                       'abs': lambda y: getattr(y, '__abs__')(),
                       'real': lambda y: getattr(y, 'real'),
                       'imag': lambda y: getattr(y, 'imag'),
                       'nanmin': lambda y: getattr(y, 'nanmin')(axis=axis),
                       'amin': lambda y: getattr(y, 'amin')(axis=axis),
                       'nanmax': lambda y: getattr(y, 'nanmax')(axis=axis),
                       'amax': lambda y: getattr(y, 'amax')(axis=axis),
                       'median': lambda y: getattr(y, 'median')(axis=axis),
                       'mean': lambda y: getattr(y, 'mean')(axis=axis),
                       'std': lambda y: getattr(y, 'std')(axis=axis),
                       'var': lambda y: getattr(y, 'var')(axis=axis),
                       'argmin_nonflat': lambda y: getattr(y,
                                                           'argmin_nonflat')(
                                                               axis=axis),
                       'argmin': lambda y: getattr(y, 'argmin')(axis=axis),
                       'argmax_nonflat': lambda y: getattr(y,
                                                           'argmax_nonflat')(
                                                               axis=axis),
                       'argmax': lambda y: getattr(y, 'argmax')(axis=axis),
                       'conjugate': lambda y: getattr(y, 'conjugate')(),
                       'sum': lambda y: getattr(y, 'sum')(axis=axis),
                       'prod': lambda y: getattr(y, 'prod')(axis=axis),
                       'unique': lambda y: getattr(y, 'unique')(),
                       'copy': lambda y: getattr(y, 'copy')(),
                       'copy_empty': lambda y: getattr(y, 'copy_empty')(),
                       'isnan': lambda y: getattr(y, 'isnan')(),
                       'isinf': lambda y: getattr(y, 'isinf')(),
                       'isfinite': lambda y: getattr(y, 'isfinite')(),
                       'nan_to_num': lambda y: getattr(y, 'nan_to_num')(),
                       'all': lambda y: getattr(y, 'all')(axis=axis),
                       'any': lambda y: getattr(y, 'any')(axis=axis),
                       'None': lambda y: y}

        return translation[op](x, **kwargs)

    def _binary_operation(self, x, y, op='None', cast=0):

        translation = {'add': lambda z: getattr(z, '__add__'),
                       'radd': lambda z: getattr(z, '__radd__'),
                       'iadd': lambda z: getattr(z, '__iadd__'),
                       'sub': lambda z: getattr(z, '__sub__'),
                       'rsub': lambda z: getattr(z, '__rsub__'),
                       'isub': lambda z: getattr(z, '__isub__'),
                       'mul': lambda z: getattr(z, '__mul__'),
                       'rmul': lambda z: getattr(z, '__rmul__'),
                       'imul': lambda z: getattr(z, '__imul__'),
                       'div': lambda z: getattr(z, '__div__'),
                       'rdiv': lambda z: getattr(z, '__rdiv__'),
                       'idiv': lambda z: getattr(z, '__idiv__'),
                       'pow': lambda z: getattr(z, '__pow__'),
                       'rpow': lambda z: getattr(z, '__rpow__'),
                       'ipow': lambda z: getattr(z, '__ipow__'),
                       'ne': lambda z: getattr(z, '__ne__'),
                       'lt': lambda z: getattr(z, '__lt__'),
                       'le': lambda z: getattr(z, '__le__'),
                       'eq': lambda z: getattr(z, '__eq__'),
                       'ge': lambda z: getattr(z, '__ge__'),
                       'gt': lambda z: getattr(z, '__gt__'),
                       'None': lambda z: lambda u: u}

        if (cast & 1) != 0:
            x = self.cast(x)
        if (cast & 2) != 0:
            y = self.cast(y)

        return translation[op](x)(y)

    def __add__(self, other):
        return self._binary_helper(other, op='add')

    def __radd__(self, other):
        return self._binary_helper(other, op='radd')

    def __iadd__(self, other):
        return self._binary_helper(other, op='iadd', inplace=True)

    def __sub__(self, other):
        return self._binary_helper(other, op='sub')

    def __rsub__(self, other):
        return self._binary_helper(other, op='rsub')

    def __isub__(self, other):
        return self._binary_helper(other, op='isub', inplace=True)

    def __mul__(self, other):
        return self._binary_helper(other, op='mul')

    def __rmul__(self, other):
        return self._binary_helper(other, op='rmul')

    def __imul__(self, other):
        return self._binary_helper(other, op='imul', inplace=True)

    def __div__(self, other):
        return self._binary_helper(other, op='div')

    def __rdiv__(self, other):
        return self._binary_helper(other, op='rdiv')

    def __idiv__(self, other):
        return self._binary_helper(other, op='idiv', inplace=True)

    def __pow__(self, other):
        return self._binary_helper(other, op='pow')

    def __rpow__(self, other):
        return self._binary_helper(other, op='rpow')

    def __ipow__(self, other):
        return self._binary_helper(other, op='ipow', inplace=True)

    def __lt__(self, other):
        return self._binary_helper(other, op='lt')

    def __le__(self, other):
        return self._binary_helper(other, op='le')

    def __ne__(self, other):
        if other is None:
            return True
        else:
            return self._binary_helper(other, op='ne')

    def __eq__(self, other):
        if other is None:
            return False
        else:
            return self._binary_helper(other, op='eq')

    def __ge__(self, other):
        return self._binary_helper(other, op='ge')

    def __gt__(self, other):
        return self._binary_helper(other, op='gt')


class EmptyField(Field):
    def __init__(self):
        pass
