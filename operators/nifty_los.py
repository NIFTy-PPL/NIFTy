# -*- coding: utf-8 -*-

import numpy as np

from line_integrator import multi_integrator, \
                            gaussian_error_function

from nifty.keepers import about,\
                          global_dependency_injector as gdi
from nifty.nifty_mpi_data import distributed_data_object,\
                                 STRATEGIES
from nifty.nifty_core import point_space,\
                             field
from nifty.rg import rg_space
from nifty.operators import operator

MPI = gdi['MPI']

class los_response(operator):

    def __init__(self, domain, starts, ends, sigmas_low=None, sigmas_up=None,
                 zero_point=None, error_function=gaussian_error_function,
                 target=None):

        if not isinstance(domain, rg_space):
            raise TypeError(about._errors.cstring(
                "ERROR: The domain must be a rg_space instance."))
        self.domain = domain
        self.codomain = self.domain.get_codomain()

        if callable(error_function):
            self.error_function = error_function
        else:
            raise ValueError(about._errors.cstring(
                "ERROR: error_function must be callable."))

        (self.starts,
         self.ends,
         self.sigmas_low,
         self.sigmas_up,
         self.zero_point) = self._parse_coordinates(self.domain,
                                                    starts, ends, sigmas_low,
                                                    sigmas_up, zero_point)

        self.local_weights_and_indices = self._compute_weights_and_indices()

        self.number_of_los = len(self.sigmas_low)

        if target is None:
            self.target = point_space(num=self.number_of_los,
                                      dtype=self.domain.dtype,
                                      datamodel=self.domain.datamodel,
                                      comm=self.domain.comm)
        else:
            self.target = target

        self.cotarget = self.target.get_codomain()

        self.imp = True
        self.uni = False
        self.sym = False

    def _parse_coordinates(self, domain, starts, ends, sigmas_low, sigmas_up,
                           zero_point):
        # basic sanity checks
        if not isinstance(starts, list):
            raise TypeError(about._errors.cstring(
                "ERROR: starts must be a list instance."))
        if not isinstance(ends, list):
            raise TypeError(about._errors.cstring(
                "ERROR: ends must be a list instance."))
        if not (len(domain.get_shape()) == len(starts) == len(ends)):
            raise ValueError(about._errors.cstring(
                "ERROR: The length of starts and ends must " +
                "be the same as the number of dimension of the domain."))

        number_of_dimensions = len(starts)

        if zero_point is None:
            zero_point = [0.] * number_of_dimensions

        if np.shape(zero_point) != (number_of_dimensions,):
            raise ValueError(about._errors.cstring(
                "ERROR: The shape of zero_point must match the length of " +
                "the starts and ends list"))
        parsed_zero_point = list(zero_point)

        # extract the number of line-of-sights and by the way check that
        # all entries of starts and ends have the right shape
        number_of_los = None
        for i in xrange(2*number_of_dimensions):
            if i < number_of_dimensions:
                temp_entry = starts[i]
            else:
                temp_entry = ends[i-number_of_dimensions]

            if isinstance(temp_entry, np.ndarray):
                if len(np.shape(temp_entry)) != 1:
                    raise ValueError(about._errors.cstring(
                        "ERROR: The numpy ndarrays in starts " +
                        "and ends must be flat."))

                if number_of_los is None:
                    number_of_los = len(temp_entry)
                elif number_of_los != len(temp_entry):
                    raise ValueError(about._errors.cstring(
                        "ERROR: The length of all numpy ndarrays in starts " +
                        "and ends must be the same."))
            elif np.isscalar(temp_entry):
                pass
            else:
                raise TypeError(about._errors.cstring(
                    "ERROR: The entries of starts and ends must be either " +
                    "scalar or numpy ndarrays."))

        if number_of_los is None:
            number_of_los = 1
            starts = [np.array([x]) for x in starts]
            ends = [np.array([x]) for x in ends]

        # Parse the coordinate arrays/scalars in the starts and ends list
        parsed_starts = self._parse_startsends(starts, number_of_los)
        parsed_ends = self._parse_startsends(ends, number_of_los)

        # check that sigmas_up/lows have the right shape and parse scalars
        parsed_sigmas_low = self._parse_sigmas_uplows(sigmas_low,
                                                      number_of_los)
        parsed_sigmas_up = self._parse_sigmas_uplows(sigmas_up, number_of_los)
        return (parsed_starts, parsed_ends, parsed_sigmas_low,
                parsed_sigmas_up, parsed_zero_point)

    def _parse_startsends(self, coords, number_of_los):
        result_coords = [None]*len(coords)
        for i in xrange(len(coords)):
            temp_array = np.empty(number_of_los, dtype=np.float)
            temp_array[:] = coords[i]
            result_coords[i] = temp_array
        return result_coords

    def _parse_sigmas_uplows(self, sig, number_of_los):
        if sig is None:
            parsed_sig = np.zeros(number_of_los, dtype=np.float)
        elif isinstance(sig, np.ndarray):
            if np.shape(sig) != (number_of_los,):
                    raise ValueError(about._errors.cstring(
                        "ERROR: The length of sigmas_up/sigmas_low must be " +
                        " the same as the number of line-of-sights."))
            parsed_sig = sig.astype(np.float)
        elif np.isscalar(sig):
            parsed_sig = np.empty(number_of_los, dtype=np.float)
            parsed_sig[:] = sig
        else:
            raise TypeError(about._errors.cstring(
                "ERROR: sigmas_up/sigmas_low must either be a scalar or a " +
                "numpy ndarray."))
        return parsed_sig

    def convert_physical_to_indices(self, physical_positions):
        pixel_coordinates = [None]*len(physical_positions)
        local_zero_point = self._get_local_zero_point()

        for i in xrange(len(pixel_coordinates)):
            # Compute the distance to the zeroth pixel.
            # Then rescale the coordinates to the uniform grid.
            pixel_coordinates[i] = ((physical_positions[i] -
                                     local_zero_point[i]) /
                                    self.domain.distances[i]) + 0.5

        return pixel_coordinates

    def _convert_physical_to_pixel_lengths(self, lengths, starts, ends):
        directions = np.array(ends) - np.array(starts)
        distances = np.array(self.domain.distances)[:, None]
        rescalers = (np.linalg.norm(directions / distances, axis=0) /
                     np.linalg.norm(directions, axis=0))
        return lengths * rescalers

    def _convert_sigmas_to_physical_coordinates(self, starts, ends,
                                                sigmas_low, sigmas_up):
        starts = np.array(starts)
        ends = np.array(ends)
        c = ends - starts
        abs_c = np.linalg.norm(c, axis=0)
        sigmas_low_coords = list(starts + (abs_c - sigmas_low)*c/abs_c)
        sigmas_up_coords = list(starts + (abs_c + sigmas_up)*c/abs_c)
        return (sigmas_low_coords, sigmas_up_coords)

    def _get_local_zero_point(self):
        if self.domain.datamodel == 'np':
            return self.zero_point
        elif self.domain.datamodel in STRATEGIES['not']:
            return self.zero_point
        elif self.domain.datamodel in STRATEGIES['slicing']:
            dummy_d2o = distributed_data_object(
                                global_shape=self.domain.get_shape(),
                                dtype=np.dtype('int16'),
                                distribution_strategy=self.domain.datamodel,
                                skip_parsing=True)

            pixel_offset = dummy_d2o.distributor.local_start
            distance_offset = pixel_offset * self.domain.distances[0]
            local_zero_point = self.zero_point[:]
            local_zero_point[0] += distance_offset
            return local_zero_point
        else:
            raise NotImplementedError(about._errors.cstring(
                "ERROR: The space's datamodel is not supported:" +
                str(self.domain.datamodel)))

    def _get_local_shape(self):
        if self.domain.datamodel == 'np':
            return self.domain.get_shape()
        elif self.domain.datamodel in STRATEGIES['not']:
            return self.domain.get_shape()
        elif self.domain.datamodel in STRATEGIES['slicing']:
            dummy_d2o = distributed_data_object(
                                global_shape=self.domain.get_shape(),
                                dtype=np.dtype('int'),
                                distribution_strategy=self.domain.datamodel,
                                skip_parsing=True)
            return dummy_d2o.distributor.local_shape

    def _compute_weights_and_indices(self):
        # compute the local pixel coordinates for the starts and ends
        localized_pixel_starts = self.convert_physical_to_indices(self.starts)
        localized_pixel_ends = self.convert_physical_to_indices(self.ends)

        # Convert the sigmas from physical distances to pixel coordinates
        # Therefore transform the distances to physical coordinates...
        (sigmas_low_coords, sigmas_up_coords) = \
            self._convert_sigmas_to_physical_coordinates(self.starts,
                                                         self.ends,
                                                         self.sigmas_low,
                                                         self.sigmas_up)
        # ...and then transform them to pixel coordinates
        localized_pixel_sigmas_low = self.convert_physical_to_indices(
                                                             sigmas_low_coords)
        localized_pixel_sigmas_up = self.convert_physical_to_indices(
                                                             sigmas_up_coords)

        # get the shape of the local data slice
        local_shape = self._get_local_shape()
        # let the cython function do the hard work of integrating over
        # the individual lines
        local_indices_and_weights_list = multi_integrator(
                                                  localized_pixel_starts,
                                                  localized_pixel_ends,
                                                  localized_pixel_sigmas_low,
                                                  localized_pixel_sigmas_up,
                                                  local_shape,
                                                  list(self.domain.distances),
                                                  self.error_function)
        return local_indices_and_weights_list

    def _multiply(self, input_field):
        # extract the local data array from the input field
        try:
            local_input_data = input_field.val.data
        except AttributeError:
            local_input_data = input_field.val

        local_result = np.zeros(self.number_of_los, dtype=self.target.dtype)

        for i in xrange(len(self.local_weights_and_indices)):
            current_weights_and_indices = self.local_weights_and_indices[i]
            (los_index, indices, weights) = current_weights_and_indices
            local_result[los_index] += \
                np.sum(local_input_data[indices]*weights)

        if self.domain.datamodel == 'np':
            global_result = local_result
        elif self.domain.datamodel is STRATEGIES['not']:
            global_result = local_result
        if self.domain.datamodel in STRATEGIES['slicing']:
            global_result = np.empty_like(local_result)
            self.domain.comm.Allreduce(local_result, global_result, op=MPI.SUM)

        result_field = field(self.target, val=global_result)
        return result_field

    def _adjoint_multiply(self, input_field):
        # get the full data as np.ndarray from the input field
        try:
            full_input_data = input_field.val.get_full_data()
        except AttributeError:
            full_input_data = input_field.val

        # produce a data_object suitable to domain
        global_result_data_object = self.domain.cast(0)

        # set the references to the underlying np arrays
        try:
            local_result_data = global_result_data_object.data
        except AttributeError:
            local_result_data = global_result_data_object

        for i in xrange(len(self.local_weights_and_indices)):
            current_weights_and_indices = self.local_weights_and_indices[i]
            (los_index, indices, weights) = current_weights_and_indices
            local_result_data[indices] += \
                (full_input_data[los_index]*weights)

        # weight the result
        local_result_data /= self.domain.get_vol()

        # construct the result field
        result_field = field(self.domain)
        try:
            result_field.val.data = local_result_data
        except AttributeError:
            result_field.val = local_result_data

        return result_field












