from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from ray.rllib.models.model import Model
from ray.rllib.models.misc import normc_initializer


class VisionNetwork(Model):

    def _init(self, inputs, num_outputs, options):
        filters = options.get("conv_filters", [
            [16, [8, 8], 4],
            [32, [4, 4], 2],
            [512, [10, 10], 1],
        ])
        fcnet_activation = options.get("fcnet_activation", "tanh")
        if num_outputs > 1:
            label_ouput = "z"
        else:
            label_ouput = "vf"

        print("those are the filters")
        print(filters)
        print("label_output")
        print(label_ouput)


        if fcnet_activation == "tanh":
            activation = tf.nn.tanh
        elif fcnet_activation == "relu":
            activation = tf.nn.relu

        with tf.name_scope("vision_net"):
            for i, (out_size, kernel, stride) in enumerate(filters[:-1], 1):
                inputs = slim.conv2d(
                    inputs, out_size, kernel, stride,
                    scope="conv{}".format(i))
            out_size, kernel, stride = filters[-1]
            fc1 = slim.conv2d(
                inputs, out_size, kernel, stride, padding="VALID", scope="fc1")

            last_layer = tf.squeeze(fc1, [1, 2])


            return slim.fully_connected(
                last_layer, num_outputs,
                weights_initializer=normc_initializer(1.0),
                activation_fn=activation,
                scope=label_ouput), last_layer