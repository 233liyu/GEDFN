
from __future__ import print_function
import numpy as np
import tensorflow as tf


class param_helper(object):
    def __init__(self, hidden_setting):
        self['layers'] = hidden_setting['layers']

    def get_weights(self, index):
        assert next(
            (item for item in self['layers'] if item["name"] == index), False)
        if self.layers:
            pass

    def return_layer_index(self):
        # return layer name
        rtn_list = []
        for item in self.layers:
            rtn_list.append(item.name)
        return rtn_list

    class iter(object):
        def __init__(self, helper, start=0):
            self.num = start
            self.helper = helper

        def __iter__(self):
            return self

        def __next__(self):
            num = self.num
            self.num += 1
            return num

    def param_select(self, partition, param):
        cp_partition = partition

        degree_sum = np.sum(partition, axis=0)
        sort_index = np.argsort(degree_sum)[::-1]

        n_node = partition.shape[0]

        for node_index in range(param, n_node):
            cp_partition[node_index] = np.zeros(n_node)
            cp_partition[:, node_index] = np.zeros(n_node)

        print(cp_partition)
        return cp_partition


class IterObj:

    def __init__(self):
        self.a = [3, 5, 7, 11, 13, 17, 19]

        self.n = len(self.a)
        self.i = 0

    def __iter__(self):
        return iter(self.a)

    def __next__(self):
        while self.i < self.n:
            v = self.a[self.i]
            self.i += 1
            return v
        else:
            self.i = 0
            raise StopIteration()
