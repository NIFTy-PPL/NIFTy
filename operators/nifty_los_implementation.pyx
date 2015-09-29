# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
cimport cython


class los_integrator(object):
    def __init__(self, shape, start, end):
        self.dtype = np.dtype('float')

        self.shape = tuple(shape)
        self.start = np.array(start, dtype=self.dtype)
        self.end = np.array(end, dtype=self.dtype)

        assert(np.all(np.array(self.shape) != 0))
        assert(len(self.shape) == len(self.start) == len(self.end))
        assert(len(self.start.shape) == len(self.end.shape) == 1)

    def integrate(self):
        if np.all(self.start == self.end):
            return self._empty_results()
        try:
            projected_start = self._project_to_cuboid(mode='start')
            projected_end = self._project_to_cuboid(mode='end')
        except ValueError:
            return self._empty_results()

        (indices, weights) = self._integrate_through_cuboid(projected_start,
                                                            projected_end)
        return (indices, weights)

    def _empty_results(self):
        return ([np.array([], dtype=np.dtype('int'))] * len(self.shape),
                np.array([], dtype=np.dtype('float')))

    def _project_to_cuboid(self, mode):
        if mode == 'start':
            a = self.start
            b = self.end
        elif mode == 'end':
            a = self.end
            b = self.start
        else:
            raise ValueError

        if np.all(np.zeros_like(a) <= a) and np.all(a <= np.array(self.shape)):
            return a

        c = b - a

        surface_list = []
        for i in xrange(len(self.shape)):
            surface_list += [[i, 0]]
            surface_list += [[i, self.shape[i]]]

        translator_list = map(lambda z:
                              self._get_translator_to_surface(a, c, *z),
                              surface_list)
        # sort the translators according to their norm, save the sorted indices
        translator_index_list = np.argsort(map(np.linalg.norm,
                                               translator_list))
        # iterate through the indices -from short to long translators- and
        # take the first translator which brings a to the actual surface of
        # the cuboid and not just to one of the parallel planes
        found = False
        for i in translator_index_list:
            p = a + translator_list[i]
            if np.all(np.zeros_like(p) <= p) and \
                    np.all(p <= np.array(self.shape)):
                found = True
                break

        if not found:
            raise ValueError(
                "ERROR: Line-of-sight does not go through cuboid.")

        return p

    def _get_translator_to_surface(self, point, full_direction,
                                   dimension_index, surface):
        """
        translates 'point' along the vector 'direction' such that the
        dimension with index 'dimension_index' has the value 'surface'
        """
        direction_scaler = np.divide((surface - point[dimension_index]),
                                     full_direction[dimension_index])
        if direction_scaler < 0 or direction_scaler > 1:
            return point * np.nan

        scaled_direction = full_direction * direction_scaler
        return scaled_direction

    def _integrate_through_cuboid(self, start, end):
        # estimate the maximum number of cells that could be hit
        # the current estimator is: norm of the vector times number of dims
        num_estimate = np.ceil(np.linalg.norm(end - start))*len(start)

        index_list = np.empty((num_estimate, len(start)),
                              dtype=np.dtype('int'))
        weight_list = np.empty((num_estimate), self.dtype)

        current_position = start
        i = 0
        while True:
            next_position, weight = self._get_next_position(current_position,
                                                            end)
            floor_current_position = np.floor(current_position)
            index_list[i] = floor_current_position
            weight_list[i] = weight

            if np.all(np.floor(current_position) == np.floor(end)):
                break

            current_position = next_position
            i += 1

        return list(index_list[:i].T), weight_list[:i]

    def _get_next_position(self, position, end_position):
        full_direction = end_position - position

        surface_list = []
        for i in xrange(len(position)):
            surface_list += [[i, strong_floor(position[i])]]
            surface_list += [[i, strong_ceil(position[i])]]

        translator_list = map(lambda z: self._get_translator_to_surface(
                                                              position,
                                                              full_direction,
                                                              *z),
                              surface_list)

#        index_of_best_translator = np.argsort(np.linalg.norm(translator_list,
#                                                             axis=1))[0]
        translator_list = np.array(translator_list)
        index_of_best_translator = np.linalg.norm(translator_list, axis=1)
        index_of_best_translator = np.argsort(index_of_best_translator)[0]


        best_translator = translator_list[index_of_best_translator]
        # if the surounding surfaces are not reachable, it must be the case
        # that the current position is in the same cell as the endpoint
        if np.isnan(np.linalg.norm(best_translator)):
            floor_position = np.floor(position)
            # check if position is in the same cell as the endpoint
            assert(np.all(floor_position == np.floor(end_position)))
            weight = np.linalg.norm(end_position - position)
            next_position = None
        else:
            next_position = position + best_translator
            weight = np.linalg.norm(best_translator)
        return (next_position, weight)


def strong_floor(x):
    floor_x = np.floor(x)
    if floor_x == x:
        return floor_x-1
    else:
        return floor_x


def strong_ceil(x):
    ceil_x = np.ceil(x)
    if ceil_x == x:
        return ceil_x+1
    else:
        return ceil_x



































