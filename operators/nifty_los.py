# -*- coding: utf-8 -*-

import numpy as np
from line_integrator import line_integrator

from nifty.keepers import about
from nifty.rg import rg_space
from nifty.operators import operator


class los_response(operator):

    def __init__(self, domain, starts, ends, sigmas_low=None, sigmas_up=None,
                 zero_point=None, error_function=lambda x: 0.5):
        if not isinstance(domain, rg_space):
            raise TypeError(about._errors.cstring(
                "ERROR: The domain must be a rg_space instance."))
        self.domain = domain

        if not callable(error_function):
            raise ValueError(about._errors.cstring(
                "ERROR: error_function must be callable."))

        (self.starts,
         self.ends,
         self.sigmas_low,
         self.sigmas_up,
         self.zero_point) = self._parse_coordinates(self.domain,
                                                    starts, ends, sigmas_low,
                                                    sigmas_up, zero_point)

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
        parsed_sigmas_up = self._parse_sigmas_uplows(sigmas_up, number_of_los)
        parsed_sigmas_low = self._parse_sigmas_uplows(sigmas_low,
                                                      number_of_los)

        return (parsed_starts, parsed_ends, parsed_sigmas_up,
                parsed_sigmas_low, parsed_zero_point)

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

    def convert_indices_to_physical(self, pixel_coordinates):
        # first of all, compute the phyiscal distance of the given pixel
        # from the zeroth-pixel
        phyiscal_distance = np.array(pixel_coordinates) * \
                            np.array(self.domain.distances)
        # add the offset of the zeroth pixel with respect to the coordinate
        # system
        physical_position = phyiscal_distance + np.array(self.zero_point)
        return physical_position.tolist()

    def convert_physical_to_indices(self, physical_position):
        # compute the distance to the zeroth pixel
        relative_position = np.array(physical_position) - \
                            np.array(self.zero_point)
        # rescale the coordinates to the uniform grid
        pixel_coordinates = relative_position / np.array(self.domain.distances)
        return pixel_coordinates.tolist()































