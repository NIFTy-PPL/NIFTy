# -*- coding: utf-8 -*-
#cython: nonecheck=False
#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np
cimport cython

FLOAT = np.float
ctypedef np.float_t FLOAT_t

INT = np.int
ctypedef np.int_t INT_t

cdef extern from "numpy/npy_math.h":
    bint isnan(double x)
    bint signbit(double x)
    double ceil(double x)
    double floor(double x)
    double sqrt(double x)

cdef FLOAT_t NAN = float("NaN")


cdef class line_integrator(object):
    cdef tuple shape
    cdef list start
    cdef list end
#    cdef FLOAT_t [:] start
#    cdef FLOAT_t [:] end

    def __init__(self, shape, start, end):
        self.shape = tuple(shape)
        self.start = list(start)
        self.end = list(end)
        assert(np.all(np.array(self.shape) != 0))
        assert(len(self.shape) == len(self.start) == len(self.end))

    cpdef tuple integrate(self):
        if list_equal_Q(self.start, self.end):
            return self._empty_results()
        try:
            projected_start = self._project_to_cuboid('start')
            projected_end = self._project_to_cuboid('end')
        except ValueError:
            return self._empty_results()

        (indices, weights) = self._integrate_through_cuboid(projected_start,
                                                            projected_end)
        return (indices, weights)

    def _empty_results(self):
        return ([np.array([], dtype=INT)] * len(self.shape),
                np.array([], dtype=FLOAT))

    cpdef list _project_to_cuboid(self, str mode):
        cdef list a, b, c, p, surface_list, translator_list,\
                  translator_index_list
        cdef int ndim, i, s1, s2
        cdef bint found

        if mode == 'start':
            a = self.start
            b = self.end
        elif mode == 'end':
            a = self.end
            b = self.start
        else:
            raise ValueError

        if list_all_le([0]*len(a), a) and list_all_le(a, list(self.shape)):
            return a

        c = list_sub(b, a)

        ndim = len(self.shape)
        surface_list = [None]*2*ndim
        for i in xrange(ndim):
            surface_list[2*i] = [i, 0]
            surface_list[2*i+1] = [i, self.shape[i]]

        translator_list = []

        for s1, s2 in surface_list:
            translator_list += [self._get_translator_to_surface(a, c,
                                                                s1, s2)]
        # sort the translators according to their norm, save the sorted indices
        translator_index_list = np.argsort(np.linalg.norm(translator_list,
                                                          axis=1)).tolist()
        # iterate through the indices -from short to long translators- and
        # take the first translator which brings a to the actual surface of
        # the cuboid and not just to one of the parallel planes
        found = False
        for i in translator_index_list:
            p = list_add(a, translator_list[i])
            if list_all_le([0]*len(p), p) and list_all_le(p, list(self.shape)):
                found = True
                break

        if not found:
            raise ValueError(
                "ERROR: Line-of-sight does not go through cuboid.")

        return p

    cdef list _get_translator_to_surface(self,
                                   list point,
                                   list full_direction,
                                   int dimension_index,
                                   int surface):
        """
        translates 'point' along the vector 'direction' such that the
        dimension with index 'dimension_index' has the value 'surface'
        """
        cdef int ndim = len(point)
        cdef list scaled_direction = [None] * ndim
        cdef FLOAT_t point_i = point[dimension_index]
        cdef FLOAT_t full_direction_i = full_direction[dimension_index]

        if full_direction_i == 0:
            return [NAN]*ndim

        cdef FLOAT_t direction_pre_scaler = surface - point_i

       # here gets checked if the direction_scaler shows in the same direction
        # and is shorter or of equal length as the full_direction_i.
        # The implementation avoids divisions in order to exclude errors
        # from numerical noise
        if ((abs(direction_pre_scaler) > abs(full_direction_i)) or
            signbit(direction_pre_scaler) != signbit(full_direction_i)):
            return [NAN]*ndim

        for i in xrange(ndim):
            # first multiply, then divide! Otherwise numerical noise will
            # produce something like: 1003.*(1./1003.) != 1.
            scaled_direction[i] = ((full_direction[i] * direction_pre_scaler)/
                                   full_direction_i)
        return scaled_direction

    cdef tuple _integrate_through_cuboid(self, list start, list end):
        cdef INT_t i, j, num_estimate
        cdef list current_position, next_position, floor_current_position
        cdef FLOAT_t weight

        # estimate the maximum number of cells that could be hit
        # the current estimator is: norm of the vector times number of dims
        num_estimate = INT(ceil(list_norm(list_sub(end, start))))*len(start)

        cdef np.ndarray[INT_t, ndim=2] index_list = np.empty((num_estimate,
                                                              len(start)),
                                                             dtype=INT)
        cdef np.ndarray[FLOAT_t, ndim=1] weight_list = np.empty(num_estimate,
                                                                FLOAT)


        current_position = start
        i = 0
        while True:
            next_position, weight = self._get_next_position(current_position,
                                                            end)
            floor_current_position = list_floor(current_position)
            for j in xrange(len(start)):
                index_list[i, j] = floor_current_position[j]
            weight_list[i] = weight

            if floor_current_position == list_floor(end):
                break

            current_position = next_position
            i += 1
        return (list(index_list[:i].T), weight_list[:i])


    cdef tuple _get_next_position(self,
                                  list position,
                                  list end_position):

        cdef list surface_list, translator_list
        cdef INT_t i, s1, s2, n_surfaces
        cdef FLOAT_t weight, best_translator_norm, temp_translator_norm
        cdef list full_direction, best_translator, temp_translator,\
                  floor_position, next_position


        full_direction = list_sub(end_position, position)

        n_surfaces = len(position)
        surface_list = [None] * n_surfaces
        for i in xrange(n_surfaces):
            if signbit(full_direction[i]):
                surface_list[i] = [i, strong_floor(position[i])]
            else:
                surface_list[i] = [i, strong_ceil(position[i])]

        best_translator_norm = NAN
        best_translator = [NAN] * len(position)
        for s1, s2 in surface_list:
            temp_translator = self._get_translator_to_surface(position,
                                                              full_direction,
                                                              s1, s2)
            temp_translator_norm = list_norm(temp_translator)
            if ((not best_translator_norm <= temp_translator_norm) and
                    (not isnan(temp_translator_norm))):
                best_translator_norm = temp_translator_norm
                best_translator = temp_translator
        # if the surounding surfaces are not reachable, it must be the case
        # that the current position is in the same cell as the endpoint
        if isnan(best_translator_norm):
            floor_position = list_floor(position)
            # check if position is in the same cell as the endpoint
            assert(floor_position == list_floor(end_position))
            weight = list_norm(list_sub(end_position, position))
            next_position = position
        else:
            next_position = list_add(position, best_translator)
            weight = list_norm(best_translator)
        return (next_position, weight)


cdef INT_t strong_floor(FLOAT_t x):
    cdef FLOAT_t floor_x
    floor_x = floor(x)
    if floor_x == x:
        return INT(floor_x - 1)
    else:
        return INT(floor_x)

cpdef INT_t strong_ceil(FLOAT_t x):
    cdef FLOAT_t ceil_x
    ceil_x = ceil(x)
    if ceil_x == x:
        return INT(ceil_x + 1)
    else:
        return INT(ceil_x)

cdef list list_floor(list l):
    cdef unsigned int i, ndim = len(l)
    cdef list result = [None] * ndim
    for i in xrange(ndim):
        result[i] = floor(l[i])
    return result

cdef list list_ceil(list l):
    cdef unsigned int i, ndim = len(l)
    cdef list result = [None] * ndim
    for i in xrange(ndim):
        result[i] = ceil(l[i])
    return result

cdef FLOAT_t list_norm(list l):
    cdef FLOAT_t d, result = 0.
    for d in l:
        result += d**2
    return sqrt(result)

cdef bint list_equal_Q(list list1, list list2):
    cdef unsigned int i
    for i in xrange(len(list1)):
        if list1[i] != list2[i]:
            return False
    return True

cdef bint list_contains_nan_Q(list l):
    cdef unsigned int i
    for i in xrange(len(l)):
        if isnan(l[i]):
            return True
    return False

cdef bint list_all_le(list list1, list list2):
    cdef unsigned int i
    for i in xrange(len(list1)):
        if list1[i] <= list2[i]:
            continue
        else:
            return False
    return True

cdef list list_add(list list1, list list2):
    cdef int ndim = len(list1)
    cdef list result = [None]*ndim
    for i in xrange(ndim):
        result[i] = list1[i] + list2[i]
    return result

cdef list list_sub(list list1, list list2):
    cdef int ndim = len(list1)
    cdef list result = [None]*ndim
    for i in xrange(ndim):
        result[i] = list1[i] - list2[i]
    return result


#def test2():
#    print ceil(1.5)
#    print floor(1.5)
#
#def test():
#    l = los_integrator_pure((1000,1000,1000), (-1,-1,-10), (1001, 1002, 1003))
#    for i in xrange(30000):
#        l._get_next_position([1.,1.1,1.2], [10.,10.,11.])
#
#def test3():
#    print sqrt(3)
#












