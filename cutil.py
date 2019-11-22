
from __future__ import print_function
import numpy as np
import tensorflow as tf


class layer_obj(object):
    def __init__(self, setting, previous_layer):
        self.name = setting['name']
        self.hidden_num = setting['hidden_num']
        self.set_limit = setting['limit']
        self.previous = previous_layer
        self.weight = self.get_weights(setting, previous_layer)
        self.bias = self.get_bias(setting)
        self.max_pool = False if "max_pool" not in setting else setting["max_pool"]
        self.dropout = False if "drop_out" not in setting else setting["drop_out"]

    def get_bias(self, setting):
        if "bias" in setting:
            return setting['bias']
        else:
            return tf.Variable(tf.zeros([self.hidden_num]))

    def get_weights(self, layer_setting, previous_layer):
        if "weights" in layer_setting:
            return layer_setting["weights"]
        else:
            return tf.Variable(tf.truncated_normal(shape=[previous_layer, layer_setting["hidden_num"]], stddev=0.1))

    def get_partition(self, partition):
        return param_select(partition, self.previous, self.hidden_num)


class param_helper(object):
    def __init__(self, hidden_setting):
        self.layers = []
        self.features = hidden_setting["gloable"]["n_features"]
        self.classes = hidden_setting["gloable"]["n_classes"]

        last_layer = self.features
        for layer in hidden_setting['layers']:
            self.layers.append(layer_obj(layer, last_layer))
            last_layer = layer["hidden_num"]

    def get_weights(self, index):
        assert next(
            (item for item in self.layers if item["name"] == index), False)

    def iter_layer(self):
        return iter(self.layers)

    class iter(object):
        def __init__(self, layers, start=0):
            self.num = start
            self.layers = layers
            self.n = len(layers)

        def __iter__(self):
            return self.layers

        def __next__(self):
            while self.num < self.n:
                self.num += 1
                return self.layers[self.num]
            else:
                self.i = 0
                raise StopIteration()


def param_select(partition, previous, current):

    degree_sum = np.sum(partition, axis=0)
    sort_index = np.argsort(degree_sum)[::-1]

    n_node = partition.shape[0]
    n_partition = np.zeros(previous * current).reshape(previous, current)

    select_index = sort_index[:current]

    current_index = sort_index[:previous]
    current_index.sort()

    i = 0
    for index in current_index:
        if index in select_index:
            n_partition[i] = np.ones(current)
        i += 1

    return n_partition

# TODO: intergrate into the project
def max_pool(mat):  # input {mat}rix

    def max_pool_one(instance):
        return tf.reduce_max(tf.multiply(tf.matmul(tf.reshape(instance, [n_features, 1]), tf.ones([1, n_features])), partition), axis=0)

    out = tf.map_fn(max_pool_one, mat,
                    parallel_iterations=1000, swap_memory=True)
    return out


def multilayer_perceptron(x, param, partition, keep_prob):

    layers = []
    last_layer = x

    for layer_param in param.iter_layer():
        if layer_param.set_limit:
            mod_weight = tf.multiply(layer_param.weight, layer_param.get_partition(partition))
        else:
            mod_weight = layer_param.weight

        layer = tf.add(tf.matmul(last_layer, mod_weight), layer_param.bias)
        layer = tf.nn.relu(layer)
        if layer_param.max_pool:
            layer = max_pool(layer)
        if layer_param.dropout:
            layer = tf.nn.dropout(layer, keep_prob=keep_prob)

        layers.append(layer)
        last_layer = layer

    return layers[-1]

