import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size],
        initializer=initializer)
    b = tf.get_variable(name + "/b", [size],
        initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b


def build_lstm(x, size, name, step_size):
    lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)

    c_init = np.zeros((1, lstm.state_size.c), np.float32)
    h_init = np.zeros((1, lstm.state_size.h), np.float32)
    state_init = [c_init, h_init]

    c_in = tf.placeholder(tf.float32,
            shape=[1, lstm.state_size.c],
            name='c_in')
    h_in = tf.placeholder(tf.float32,
            shape=[1, lstm.state_size.h],
            name='h_in')
    state_in = [c_in, h_in]

    state_in = rnn.LSTMStateTuple(c_in, h_in)

    lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
        lstm, x, initial_state=state_in, sequence_length=step_size,
        time_major=False)
    lstm_outputs = tf.reshape(lstm_outputs, [-1, size])

    lstm_c, lstm_h = lstm_state
    state_out = [lstm_c[:1, :], lstm_h[:1, :]]
    return lstm_outputs, state_init, state_in, state_out

"""
class SingleStepLSTM(object):

    def __init__(self,x , size, step_size):
        lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)
        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]

        c_in = tf.placeholder(tf.float32,
                shape=[1, lstm.state_size.c],
                name='c_in')
        h_in = tf.placeholder(tf.float32,
                shape=[1, lstm.state_size.h],
                name='h_in')
        self.state_in = [c_in, h_in]
        state_in = rnn.LSTMStateTuple(c_in, h_in)

        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_outputs = tf.reshape(lstm_outputs, [-1, size])

        lstm_c, lstm_h = lstm_state
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        self.output = lstm_outputs
"""

class SingleStepLSTM(object):

    def __init__(self,x , size, step_size):
        lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)
        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]

        c_in = tf.placeholder(tf.float32,
                shape=[1, lstm.state_size.c],
                name='c_in')
        h_in = tf.placeholder(tf.float32,
                shape=[1, lstm.state_size.h],
                name='h_in')
        self.state_in = [c_in, h_in]
        state_in = rnn.LSTMStateTuple(c_in, h_in)

        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_outputs = tf.reshape(lstm_outputs, [-1, size])

        lstm_c, lstm_h = lstm_state
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        self.output = lstm_outputs