from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import distutils.version

from ray.rllib.models.misc import (conv2d, linear, flatten,
                                   normc_initializer)
from ray.rllib.models.model import Model
import tensorflow.contrib.slim as slim
from ray.rllib.models.misc import normc_initializer



class LSTM(Model):
    """Vision LSTM network based here:
    https://github.com/openai/universe-starter-agent"""

    # TODO(rliaw): Implement dilated LSTM
    def _init(self, inputs, num_outputs, options):

        lstm = rnn.BasicLSTMCell(num_outputs, state_is_tuple=True)
        #dilatation_rate = options.get("dilatation_rate", 10)

        c_in = inputs[1]
        h_in = inputs[2]
        self.state_in = [c_in, h_in]

        state_in = rnn.LSTMStateTuple(c_in, h_in)
        x = tf.expand_dims(inputs[0], [0])
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=tf.shape(inputs[0])[:1],
            time_major=False)
        lstm_outputs = tf.reshape(lstm_outputs, [-1, num_outputs])

        lstm_c, lstm_h = lstm_state
        last_layer = [lstm_c[:1, :], lstm_h[:1, :]]

        if options.get("final_layer", True):
            output = slim.fully_connected(
                lstm_outputs, num_outputs,
                weights_initializer=normc_initializer(1.0),
                activation_fn=None, scope="output_layer")
        else:
            output = lstm_outputs

        return output, last_layer
