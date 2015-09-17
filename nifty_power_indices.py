# -*- coding: utf-8 -*-

import numpy as np
from nifty.keepers import about,\
                          global_configuration as gc,\
                          global_dependency_injector as gdi
from nifty.nifty_mpi_data import distributed_data_object,\
                                 STRATEGIES as DISTRIBUTION_STRATEGIES

MPI = gdi[gc['mpi_module']]
hp = gdi['healpy']


class power_indices(object):
    def __init__(self, datamodel, allowed_distribution_strategies, comm,
                 log=False, nbin=None, binbounds=None):
        """
            Returns an instance of the power_indices class. Given the shape and
            the density of a underlying rectangular grid it provides the user
            with the pindex, kindex, rho and pundex. The indices are bined
            according to the supplied parameter scheme. If wanted, computed
            results are stored for future reuse.

            Parameters
            ----------
            shape : tuple, list, ndarray
                Array-like object which specifies the shape of the underlying
                rectangular grid
            dgrid : tuple, list, ndarray
                Array-like object which specifies the step-width of the
                underlying grid
            zerocentered : boolean, tuple/list/ndarray of boolean *optional*
                Specifies which dimensions are zerocentered. (default:False)
            log : bool *optional*
                Flag specifying if the binning of the default indices is
                performed on logarithmic scale.
            nbin : integer *optional*
                Number of used bins for the binning of the default indices.
            binbounds : {list, array}
                Array-like inner boundaries of the used bins of the default
                indices.
        """
        if comm is None:
            self.comm = getattr(gdi[gc['mpi_module']], gc['default_comm'])
        else:
            self.comm = comm

        self.datamodel = datamodel
        self.allowed_distribution_strategies = allowed_distribution_strategies

        # Compute the global kdict
        self.kdict = self.compute_kdict()
        # Initialize the dictonary which stores all individual index-dicts
        self.global_dict = {}
        # Set self.default_parameters
        self.set_default(log=log, nbin=nbin, binbounds=binbounds)

    # Redirect the direct calls approaching a power_index instance to the
    # default_indices dict
    @property
    def default_indices(self):
        return self.get_index_dict(**self.default_parameters)

    def __getitem__(self, x):
        return self.default_indices.get(x)

    def __contains__(self, x):
        return self.default_indices.__contains__(x)

    def __iter__(self):
        return self.default_indices.__iter__()

    def __getattr__(self, x):
        return self.default_indices.__getattribute__(x)

    def set_default(self, **kwargs):
        """
            Sets the index-set which is specified by the parameters as the
            default for the power_index instance.

            Parameters
            ----------
            log : bool
                Flag specifying if the binning is performed on logarithmic
                scale.
            nbin : integer
                Number of used bins.
            binbounds : {list, array}
                Array-like inner boundaries of the used bins.

            Returns
            -------
            None
        """
        parsed_kwargs = self._cast_config(**kwargs)
        self.default_parameters = parsed_kwargs

    def _cast_config(self, **kwargs):
        """
            internal helper function which casts the various combinations of
            possible parameters into a properly defaulted dictionary
        """
        temp_config_dict = kwargs.get('config_dict', None)
        if temp_config_dict is not None:
            return self._cast_config_helper(**temp_config_dict)
        else:
            temp_log = kwargs.get("log", None)
            temp_nbin = kwargs.get("nbin", None)
            temp_binbounds = kwargs.get("binbounds", None)

            return self._cast_config_helper(log=temp_log,
                                            nbin=temp_nbin,
                                            binbounds=temp_binbounds)

    def _cast_config_helper(self, log, nbin, binbounds):
        """
            internal helper function which sets the defaults for the
            _cast_config function
        """

        try:
            temp_log = bool(log)
        except(TypeError):
            temp_log = False

        try:
            temp_nbin = int(nbin)
        except(TypeError):
            temp_nbin = None

        try:
            temp_binbounds = tuple(np.array(binbounds))
        except(TypeError):
            temp_binbounds = None

        temp_dict = {"log": temp_log,
                     "nbin": temp_nbin,
                     "binbounds": temp_binbounds}
        return temp_dict

    def compute_kdict(self):
        raise NotImplementedError(
            about._errors.cstring(
                "ERROR: No generic compute_kdict method implemented."))

    def get_index_dict(self, **kwargs):
        """
            Returns a dictionary containing the pindex, kindex, rho and pundex
            binned according to the supplied parameter scheme and a
            configuration dict containing this scheme.

            Parameters
            ----------
            store : bool
                Flag specifying if  the calculated index dictionary should be
                stored in the global_dict for future use.
            log : bool
                Flag specifying if the binning is performed on logarithmic
                scale.
            nbin : integer
                Number of used bins.
            binbounds : {list, array}
                Array-like inner boundaries of the used bins.

            Returns
            -------
            index_dict : dict
                Contains the keys: 'config', 'pindex', 'kindex', 'rho' and
                'pundex'
        """
        # Cast the input arguments
        temp_config_dict = self._cast_config(**kwargs)
        # Compute a hashable identifier from the config which will be used
        # as dict key
        temp_key = self._freeze_config(temp_config_dict)
        # Check if the result should be stored for future use.
        storeQ = kwargs.get("store", True)
        # Try to find the requested index dict in the global_dict
        try:
            return self.global_dict[temp_key]
        except(KeyError):
            # If it is not found, calculate it.
            temp_index_dict = self._compute_index_dict(temp_config_dict)
            # Store it, if required
            if storeQ:
                self.global_dict[temp_key] = temp_index_dict
                # Important: If the result is stored, return a reference to
                # the dictionary entry, not anly a plain copy. Otherwise,
                # set_default breaks!
                return self.global_dict[temp_key]
            else:
                # Return the plain result.
                return temp_index_dict

    def _freeze_config(self, config_dict):
        """
            a helper function which forms a hashable identifying object from
            a config dictionary which can be used as key of a dict
        """
        return frozenset(config_dict.items())

    def _compute_index_dict(self, config_dict):
        """
            Internal helper function which takes a config_dict, asks for the
            pindex/kindex/rho/pundex set, and bins them according to the config
        """
        # if no binning is requested, compute the indices, build the dict,
        # and return it straight.
        if not config_dict["log"] and config_dict["nbin"] is None and \
                config_dict["binbounds"] is None:
            (temp_pindex, temp_kindex, temp_rho, temp_pundex) =\
                self._compute_indices(self.kdict)
            temp_kdict = self.kdict

        # if binning is required, make a recursive call to get the unbinned
        # indices, bin them, compute the pundex and then return everything.
        else:
            # Get the unbinned indices
            temp_unbinned_indices = self.get_index_dict(store=False)
            # Bin them
            (temp_pindex, temp_kindex, temp_rho, temp_pundex) = \
                self._bin_power_indices(
                    temp_unbinned_indices, **config_dict)
            # Make a binned version of kdict
            temp_kdict = self._compute_kdict_from_pindex_kindex(temp_pindex,
                                                                temp_kindex)

        temp_index_dict = {"config": config_dict,
                           "pindex": temp_pindex,
                           "kindex": temp_kindex,
                           "rho": temp_rho,
                           "pundex": temp_pundex,
                           "kdict": temp_kdict}
        return temp_index_dict

    def _compute_kdict_from_pindex_kindex(self, pindex, kindex):
        if isinstance(pindex, distributed_data_object):
            tempindex = pindex.copy(dtype=kindex.dtype)
            result = tempindex.apply_scalar_function(lambda x: kindex[x])
        else:
            result = kindex[pindex].astype(dtype=kindex.dtype)
        return result

    def _compute_indices(self, nkdict):
        if self.datamodel == 'np':
            return self._compute_indices_np(nkdict)
        elif self.datamodel in self.allowed_distribution_strategies:
            return self._compute_indices_d2o(nkdict)
        else:
            raise ValueError(about._errors.cstring(
                'ERROR: Datamodel is not supported.'))

    def _compute_indices_d2o(self, nkdict):
        """
        Internal helper function which computes pindex, kindex, rho and pundex
        from a given nkdict
        """
        raise NotImplementedError(
            about._errors.cstring(
                "ERROR: No generic _compute_indices_d2o method implemented."))

    def _compute_pundex_d2o(self, global_pindex, global_kindex):
        """
        Internal helper function which computes the pundex array from a
        pindex and a kindex array. This function is separated from the
        _compute_indices function as it is needed in _bin_power_indices,
        too.
        """
        ##########
        # pundex #
        ##########
        # Prepare the local data
        local_pindex = global_pindex.get_local_data()
        # Compute the local pundices for the local pindices
        (temp_uniqued_pindex, local_temp_pundex) = np.unique(local_pindex,
                                                             return_index=True)
        # Shift the local pundices by the nodes' local_dim_offset
        local_temp_pundex += global_pindex.distributor.local_dim_offset

        # Prepare the pundex arrays used for the Allreduce operation
        # pundex has the same length as the kindex array
        local_pundex = np.zeros(shape=global_kindex.shape, dtype=np.int)
        # Set the default value higher than the maximal possible pundex value
        # so that MPI.MIN can sort out the default
        local_pundex += np.prod(global_pindex.shape) + 1
        # Set the default value higher than the length
        global_pundex = np.empty_like(local_pundex)
        # Store the individual pundices in the local_pundex array
        local_pundex[temp_uniqued_pindex] = local_temp_pundex
        # Use Allreduce to find the first occurences/smallest pundices
        self.comm.Allreduce(local_pundex, global_pundex, op=MPI.MIN)
        return global_pundex

    def _compute_indices_np(self, nkdict):
        """
        Internal helper function which computes pindex, kindex, rho and pundex
        from a given nkdict
        """
        raise NotImplementedError(
            about._errors.cstring(
                "ERROR: No generic _compute_indices_np method implemented."))

    def _compute_pundex_np(self, pindex, kindex):
        """
        Internal helper function which computes the pundex array from a
        pindex and a kindex array. This function is separated from the
        _compute_indices function as it is needed in _bin_power_indices,
        too.
        """
        ##########
        # pundex #
        ##########
        pundex = np.unique(pindex, return_index=True)[1]
        return pundex

    def _bin_power_indices(self, index_dict, **kwargs):
        """
            Returns the binned power indices associated with the Fourier grid.

            Parameters
            ----------
            pindex : distributed_data_object
                Index of the Fourier grid points in a distributed_data_object.
            kindex : ndarray
                Array of all k-vector lengths.
            rho : ndarray
                Degeneracy factor of the individual k-vectors.
            log : bool
                Flag specifying if the binning is performed on logarithmic
                scale.
            nbin : integer
                Number of used bins.
            binbounds : {list, array}
                Array-like inner boundaries of the used bins.

            Returns
            -------
            pindex : distributed_data_object
            kindex, rho, pundex : ndarrays
                The (re)binned power indices.

        """
        # Cast the given config
        temp_config_dict = self._cast_config(**kwargs)
        log = temp_config_dict['log']
        nbin = temp_config_dict['nbin']
        binbounds = temp_config_dict['binbounds']

        # Extract the necessary indices from the supplied index dict
        pindex = index_dict["pindex"]
        kindex = index_dict["kindex"]
        rho = index_dict["rho"]

        # boundaries
        if(binbounds is not None):
            binbounds = np.sort(binbounds)
        # equal binning
        else:
            if(log is None):
                log = False
            if(log):
                k = np.r_[0, np.log(kindex[1:])]
            else:
                k = kindex
            dk = np.max(k[2:] - k[1:-1])  # minimal dk
            if(nbin is None):
                nbin = int((k[-1] - 0.5 * (k[2] + k[1])) /
                           dk - 0.5)  # maximal nbin
            else:
                nbin = min(int(nbin), int(
                    (k[-1] - 0.5 * (k[2] + k[1])) / dk + 2.5))
                dk = (k[-1] - 0.5 * (k[2] + k[1])) / (nbin - 2.5)
            binbounds = np.r_[0.5 * (3 * k[1] - k[2]),
                              0.5 * (k[1] + k[2]) + dk * np.arange(nbin - 2)]
            if(log):
                binbounds = np.exp(binbounds)
        # reordering
        reorder = np.searchsorted(binbounds, kindex)
        rho_ = np.zeros(len(binbounds) + 1, dtype=rho.dtype)
        kindex_ = np.empty(len(binbounds) + 1, dtype=kindex.dtype)
        for ii in range(len(reorder)):
            if(rho_[reorder[ii]] == 0):
                kindex_[reorder[ii]] = kindex[ii]
                rho_[reorder[ii]] += rho[ii]
            else:
                kindex_[reorder[ii]] = ((kindex_[reorder[ii]] *
                                         rho_[reorder[ii]] +
                                         kindex[ii] * rho[ii]) /
                                        (rho_[reorder[ii]] + rho[ii]))
                rho_[reorder[ii]] += rho[ii]

        if self.datamodel == 'np':
            pindex_ = reorder[pindex]
            pundex_ = self._compute_pundex_np(pindex_, kindex_)
        elif self.datamodel in self.allowed_distribution_strategies:
            pindex_ = pindex.copy_empty()
            pindex_.set_local_data(reorder[pindex.get_local_data()])
            pundex_ = self._compute_pundex_d2o(pindex_, kindex_)
        else:
            raise ValueError(about._errors.cstring(
                'ERROR: Datamodel is not supported.'))

        return pindex_, kindex_, rho_, pundex_


class rg_power_indices(power_indices):

    def __init__(self, shape, dgrid, datamodel,
                 allowed_distribution_strategies,
                 zerocentered=False, log=False, nbin=None,
                 binbounds=None, comm=None):
        """
            Returns an instance of the power_indices class. Given the shape and
            the density of a underlying rectangular grid it provides the user
            with the pindex, kindex, rho and pundex. The indices are bined
            according to the supplied parameter scheme. If wanted, computed
            results are stored for future reuse.

            Parameters
            ----------
            shape : tuple, list, ndarray
                Array-like object which specifies the shape of the underlying
                rectangular grid
            dgrid : tuple, list, ndarray
                Array-like object which specifies the step-width of the
                underlying grid
            zerocentered : boolean, tuple/list/ndarray of boolean *optional*
                Specifies which dimensions are zerocentered. (default:False)
            log : bool *optional*
                Flag specifying if the binning of the default indices is
                performed on logarithmic scale.
            nbin : integer *optional*
                Number of used bins for the binning of the default indices.
            binbounds : {list, array}
                Array-like inner boundaries of the used bins of the default
                indices.
        """
        # Basic inits and consistency checks
        self.shape = np.array(shape, dtype=int)
        self.dgrid = np.abs(np.array(dgrid))
        if self.shape.shape != self.dgrid.shape:
            raise ValueError(about._errors.cstring("ERROR: The supplied shape\
                and dgrid have not the same dimensionality"))
        self.zerocentered = self._cast_zerocentered(zerocentered)

        super(rg_power_indices, self).__init__(
            datamodel=datamodel,
            allowed_distribution_strategies=allowed_distribution_strategies,
            comm=comm,
            log=log,
            nbin=nbin,
            binbounds=binbounds)

    def _cast_zerocentered(self, zerocentered=False):
        """
            internal helper function which brings the zerocentered input in
            the form of a boolean-tuple
        """
        zc = np.array(zerocentered).astype(bool)
        if zc.shape == self.shape.shape:
            return tuple(zc)
        else:
            temp = np.empty(shape=self.shape.shape, dtype=bool)
            temp[:] = zc
            return tuple(temp)

    def compute_kdict(self):
        """
            Calculates an n-dimensional array with its entries being the
            lengths of the k-vectors from the zero point of the grid.

            Parameters
            ----------
            None : All information is taken from the parent object.

            Returns
            -------
            nkdict : distributed_data_object
        """
        shape = self.shape
        if self.datamodel == 'np':
            slice_of_first_dimension = slice(0, shape[0])
            nkdict = self._compute_kdict_helper(slice_of_first_dimension)

        elif self.datamodel in self.allowed_distribution_strategies:
            # prepare the distributed_data_object
            nkdict = distributed_data_object(
                                global_shape=shape,
                                dtype=np.float128,
                                distribution_strategy=self.datamodel)
            if self.datamodel in DISTRIBUTION_STRATEGIES['slicing']:
                # get the node's individual slice of the first dimension
                slice_of_first_dimension = slice(
                                        *nkdict.distributor.local_slice[0:2])
            elif self.datamodel in DISTRIBUTION_STRATEGIES['not']:
                slice_of_first_dimension = slice(0, shape[0])
            else:
                raise ValueError(about._errors.cstring(
                    "ERROR: Unsupported distribution strategy"))
            dists = self._compute_kdict_helper(slice_of_first_dimension)
            nkdict.set_local_data(dists)

        else:
            raise ValueError(about._errors.cstring(

                "ERROR: Unsupported datamodel"))
        return nkdict

    def _compute_kdict_helper(self, slice_of_first_dimension):
        dk = self.dgrid
        shape = self.shape

        inds = []
        for a in shape:
            inds += [slice(0, a)]

        cords = np.ogrid[inds]

        dists = ((np.float128(0) + cords[0] - shape[0] // 2) * dk[0])**2
        # apply zerocenteredQ shift
        if self.zerocentered[0] == False:
            dists = np.fft.fftshift(dists)
        # only save the individual slice
        dists = dists[slice_of_first_dimension]
        for ii in range(1, len(shape)):
            temp = ((cords[ii] - shape[ii] // 2) * dk[ii])**2
            if self.zerocentered[ii] == False:
                temp = np.fft.fftshift(temp)
            dists = dists + temp
        dists = np.sqrt(dists)
        return dists

    def _compute_indices_d2o(self, nkdict):
        """
        Internal helper function which computes pindex, kindex, rho and pundex
        from a given nkdict
        """
        ##########
        # kindex #
        ##########
        global_kindex = nkdict.unique()

        ##########
        # pindex #
        ##########
        # compute the local pindex slice on basis of the local nkdict data
        local_pindex = np.searchsorted(global_kindex, nkdict.get_local_data())
        # prepare the distributed_data_object
        global_pindex = distributed_data_object(
                                        global_shape=nkdict.shape,
                                        dtype=local_pindex.dtype,
                                        distribution_strategy=self.datamodel,
                                        comm=self.comm)
        # store the local pindex data in the global_pindex d2o
        global_pindex.set_local_data(local_pindex)

        #######
        # rho #
        #######
        global_rho = global_pindex.bincount()

        ##########
        # pundex #
        ##########
        global_pundex = self._compute_pundex_d2o(global_pindex,
                                                 global_kindex)

        return global_pindex, global_kindex, global_rho, global_pundex

    def _compute_indices_np(self, nkdict):
        """
        Internal helper function which computes pindex, kindex, rho and pundex
        from a given nkdict
        """
        ##########
        # kindex #
        ##########
        kindex = np.unique(nkdict)

        ##########
        # pindex #
        ##########
        pindex = np.searchsorted(kindex, nkdict)

        #######
        # rho #
        #######
        rho = np.bincount(pindex.flatten())

        ##########
        # pundex #
        ##########
        pundex = self._compute_pundex_np(pindex, kindex)

        return pindex, kindex, rho, pundex


class lm_power_indices(power_indices):

    def __init__(self, lmax, dim, datamodel,
                 allowed_distribution_strategies,
                 zerocentered=False, log=False, nbin=None,
                 binbounds=None, comm=None):
        """
            Returns an instance of the power_indices class. Given the shape and
            the density of a underlying rectangular grid it provides the user
            with the pindex, kindex, rho and pundex. The indices are bined
            according to the supplied parameter scheme. If wanted, computed
            results are stored for future reuse.

            Parameters
            ----------
            shape : tuple, list, ndarray
                Array-like object which specifies the shape of the underlying
                rectangular grid
            dgrid : tuple, list, ndarray
                Array-like object which specifies the step-width of the
                underlying grid
            zerocentered : boolean, tuple/list/ndarray of boolean *optional*
                Specifies which dimensions are zerocentered. (default:False)
            log : bool *optional*
                Flag specifying if the binning of the default indices is
                performed on logarithmic scale.
            nbin : integer *optional*
                Number of used bins for the binning of the default indices.
            binbounds : {list, array}
                Array-like inner boundaries of the used bins of the default
                indices.
        """
        # Basic inits and consistency checks
        self.lmax = np.uint(lmax)
        self.dim = np.uint(dim)
        super(lm_power_indices, self).__init__(
            datamodel=datamodel,
            allowed_distribution_strategies=allowed_distribution_strategies,
            comm=comm,
            log=log,
            nbin=nbin,
            binbounds=binbounds)

    def compute_kdict(self):
        """
            Calculates an n-dimensional array with its entries being the
            lengths of the k-vectors from the zero point of the grid.

            Parameters
            ----------
            None : All information is taken from the parent object.

            Returns
            -------
            nkdict : distributed_data_object
        """
        if self.datamodel == 'np':
            if 'healpy' in gdi:  # default
                nkdict = hp.Alm.getlm(self.lmax, i=None)[0]
            else:
                nkdict = self._getlm()[0]

        elif self.datamodel in self.allowed_distribution_strategies:
            raise NotImplementedError

        else:
            raise ValueError(about._errors.cstring(
                "ERROR: Unsupported datamodel"))
        return nkdict

    def _getlm(self):  # > compute all (l,m)
        index = np.arange(self.dim)
        n = 2 * self.lmax + 1
        m = np.ceil((n - np.sqrt(n**2 - 8 * (index - self.lmax))) / 2
                    ).astype(np.int)
        l = index - self.lmax * m + m * (m - 1) // 2
        return l, m

    def _compute_indices_d2o(self, nkdict):
        """
        Internal helper function which computes pindex, kindex, rho and pundex
        from a given nkdict
        """
        raise NotImplementedError

    def _compute_indices_np(self, nkdict):
        """
        Internal helper function which computes pindex, kindex, rho and pundex
        from a given nkdict
        """
        ##########
        # kindex #
        ##########
        kindex = np.arange(self.lmax + 1, dtype=np.float)

        ##########
        # pindex #
        ##########
        pindex = nkdict.astype(np.int, copy=True)

        #######
        # rho #
        #######
        rho = (2 * kindex + 1).astype(np.int)

        ##########
        # pundex #
        ##########
        pundex = self._compute_pundex_np(pindex, kindex)

        return pindex, kindex, rho, pundex
