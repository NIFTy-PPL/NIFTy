# -*- coding: utf-8 -*-

import numpy as np


class d2o_iter(object):
    def __init__(self, d2o):
        self.d2o = d2o
        self.i = 0
        self.n = np.prod(self.d2o.shape)
        self.initialize_current_local_data()

    def __iter__(self):
        return self

    def next(self):
        if self.n == 0:
            raise StopIteration()

        self.update_current_local_data()
        if self.i < self.n:
            i = self.i
            self.i += 1
            return self.current_local_data[i]
        else:
            raise StopIteration()

    def initialize_current_local_data(self):
        raise NotImplementedError

    def update_current_local_data(self):
        raise NotImplementedError


class d2o_not_iter(d2o_iter):
    def initialize_current_local_data(self):
        self.current_local_data = self.d2o.data.flatten()

    def update_current_local_data(self):
        pass


class d2o_slicing_iter(d2o_iter):
    def __init__(self, d2o):
        self.d2o = d2o
        self.i = 0
        self.n = np.prod(self.d2o.shape)
        self.local_dim_offset_list = \
            self.d2o.distributor.all_local_slices[:, 4]
        self.active_node = None

        self.initialize_current_local_data()

    def initialize_current_local_data(self):
        self.update_current_local_data()

    def update_current_local_data(self):
        new_active_node = np.searchsorted(self.local_dim_offset_list,
                                          self.i,
                                          side='right')-1
        # new_active_node = min(new_active_node, self.d2o.comm.size-1)
        if self.active_node != new_active_node:
            self.active_node = new_active_node

            self.current_local_data = self.d2o.comm.bcast(
                                        self.d2o.get_local_data().flatten(),
                                        root=self.active_node)
