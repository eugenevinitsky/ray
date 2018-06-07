from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ray.rllib.models import ModelCatalog
from ray.rllib.models.singlestepLSTM import SingleStepLSTM
import numpy as np
from scipy.special import iv

SEED= 89789775

class FeudalLoss(object):


    other_output = ["vf_preds_manager", "vf_preds_worker", "logprobs"]
    is_recurrent = False

    def __init__(
            self, __g__, __gsum__, __z__, observation_space, action_space,
            observations, value_targets_worker, advantages_worker, actions,
            distribution_class_obs, config, sess, registry, ES,
            diff=None, value_targets_manager=None, advantages_manager=None):

        self.carried_g = tf.stop_gradient(__g__)
        self.carried_z = tf.stop_gradient(__z__)
        self.carried_gsum = tf.stop_gradient(__gsum__)

        self.ES = ES
        self.action_dim= action_space.n
        self.actions = actions
        # Saved so that we can compute actions given different observations
        self.observations = tf.cast(observations, tf.float32)
        self.diff = diff



        with tf.variable_scope("CNN_filter"):
            conv1 = tf.layers.conv2d(inputs=self.observations,
                                         filters=16,
                                         kernel_size=[8, 8],
                                         activation=tf.nn.elu,
                                         strides=4)
            conv2 = tf.layers.conv2d(inputs=conv1,
                                         filters=32,
                                         kernel_size=[4, 4],
                                         activation=tf.nn.elu,
                                         strides=2)


            flattened_filters = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])

        
        with tf.variable_scope("z"):
            
            self.z = tf.layers.dense(inputs=flattened_filters, \
                                         units=config["units_z"], \
                                         activation=tf.nn.relu)

        with tf.variable_scope("Goal_Designer"):
            self.s = tf.layers.dense(inputs=self.z, \
                                 units=config["g_dim"], \
                                 activation=tf.nn.elu)

            x = tf.expand_dims(self.s, [0])

            if config["LSTM_OR_NOT"]:

                self.manager_lstm = SingleStepLSTM(size=config["g_dim"], dilatation_rate=config["dilatation_rate"])
                g_hat = self.manager_lstm.compute_step(x, step_size=tf.shape(self.observations)[:1])

            else:
                g_hat = tf.layers.dense(inputs=self.s, \
                                units=config["g_dim"], \
                                activation=tf.nn.elu)


            g_hat = tf.reshape(g_hat, shape=(-1, config["g_dim"]))

            self.g = tf.nn.l2_normalize(g_hat, dim=1)


        with tf.variable_scope("Manager_VF"):
            if not config["separate_VF_approximator"]:
                hidden_manager = tf.layers.dense(inputs=self.g, \
                                                 units=config["vf_hidden_size"][-1], \
                                                 activation=tf.nn.elu)

                weights_VF_manager = tf.get_variable("weights_VF_manager", (config["vf_hidden_size"][-1], 1))
                self.value_function_manager = tf.matmul(hidden_manager, weights_VF_manager)


            else:
                conv1_vf_appro = tf.layers.conv2d(inputs=self.observations,
                                                  filters=16,
                                                  kernel_size=[8, 8],
                                                  activation=tf.nn.elu,
                                                  strides=4)
                conv2_vf_appro = tf.layers.conv2d(inputs=conv1_vf_appro,
                                                  filters=32,
                                                  kernel_size=[4, 4],
                                                  activation=tf.nn.elu,
                                                  strides=2)
                flattened_filters_vf_appro = tf.reshape(conv2_vf_appro,
                                                        [-1, np.prod(conv2_vf_appro.get_shape().as_list()[1:])])

                layer_manager = flattened_filters_vf_appro
                for size in config["vf_hidden_size"]:
                    layer_manager = tf.layers.dense(inputs=layer_manager, \
                                            units=size, \
                                            activation=tf.nn.relu)

                weights_VF_manager = tf.get_variable("weights_VF_manager", (config["vf_hidden_size"][-1], 1))
                self.value_function_manager = tf.matmul(layer_manager, weights_VF_manager)

        with tf.variable_scope("Manager_Loss"):

            self.manager_output = self.g
            self.vf_loss_manager = tf.square(self.value_function_manager - value_targets_manager)
            self.mean_vf_loss_manager = tf.reduce_mean(self.vf_loss_manager)

            if not self.ES:

                self.diff = diff
                self.diff = tf.nn.l2_normalize(self.diff, dim=1)

                #C_p = config["kappa"] ** (config["g_dim"] / 2) / ((2 * np.pi) ** (config["g_dim"] / 2) * iv(config["g_dim"] / 2 - 1, config["g_dim"]))
                #self.logp_manager = tf.cast(tf.log(C_p), tf.float32) + config["kappa"] * dot_product

                self.logp_manager = config["kappa"] * tf.reduce_sum(tf.multiply(self.diff, self.g), axis=1)

                self.surr_manager = self.logp_manager * advantages_manager
                self.mean_surr_manager = tf.reduce_mean(self.surr_manager)

                self.loss_manager = tf.reduce_mean(
                        -self.surr_manager +
                        config["vf_loss_coeff_manager"] * self.vf_loss_manager)

        with tf.variable_scope("Worker_Goal_Embedding"):
            if config["LSTM_OR_NOT"]:
                dimension_lstm_worker = self.action_dim * config["k"]
                lstm_cell_worker = tf.nn.rnn_cell.BasicLSTMCell(dimension_lstm_worker)
                initial_state = lstm_cell_worker.zero_state(1, dtype=tf.float32)
                outputs_worker, state = tf.nn.dynamic_rnn(lstm_cell_worker, tf.expand_dims(self.carried_z , [0]),
                                                    initial_state=initial_state,
                                                   dtype=tf.float32)
                outputs_worker = tf.squeeze(outputs_worker, axis=0)

            else:
                outputs_worker = tf.layers.dense(inputs=self.carried_z, \
                                    units=self.action_dim * config["k"], \
                                    activation=tf.nn.elu)

            U = tf.reshape(outputs_worker, [-1, self.action_dim, config["k"]])

            # Calculate w

            phi = tf.get_variable("phi", (config["g_dim"], config['k']))
            w = tf.matmul(self.carried_gsum, phi)
            w = tf.expand_dims(w, [2])
            # Calculate policy and sample
            self.curr_logits = tf.reshape(tf.matmul(U, w), [-1, self.action_dim])
            self.pi = tf.nn.softmax(self.curr_logits)
            self.log_pi = tf.nn.log_softmax(self.curr_logits)

            def categorical_sample(logits, d):
                value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(
                    logits, [1], keep_dims=True), 1), [1])
                return tf.one_hot(value, d)

            self.sampler = categorical_sample(
                tf.reshape(self.curr_logits, [-1, self.action_dim]), self.action_dim)[0, :]

        with tf.variable_scope("Worker_VF"):
            if not config["separate_VF_approximator"]:
                hidden_VF_worker = tf.layers.dense(inputs=outputs_worker, \
                                         units=config["vf_hidden_size"][-1], \
                                         activation=tf.nn.elu)

                weights_VF_worker = tf.get_variable("weights_VF_worker", (config["vf_hidden_size"][-1], self.action_dim))
                self.value_function_worker = tf.matmul(hidden_VF_worker, weights_VF_worker)
            else:
                layer_worker = self.carried_g
                for size in config["vf_hidden_size"]:
                    layer_worker = tf.layers.dense(inputs=layer_worker, \
                                            units=size, \
                                            activation=tf.nn.relu)

                weights_VF_worker = tf.get_variable("weights_VF_worker",
                                                    (config["vf_hidden_size"][-1], self.action_dim))
                self.value_function_worker = tf.matmul(layer_worker, weights_VF_worker)


        with tf.variable_scope("Worker_Loss"):

            self.entropy_worker = -tf.reduce_sum(self.pi * self.log_pi)
            self.mean_entropy_worker = tf.reduce_mean(self.entropy_worker)

            self.vf_loss_worker = tf.square(self.value_function_worker * self.actions - value_targets_worker)
            self.mean_vf_loss_worker = tf.reduce_mean(self.vf_loss_worker)

            self.surr_worker = tf.reduce_sum(self.log_pi * advantages_worker, [1])
            self.mean_policy_loss_worker = -tf.reduce_mean(self.surr_worker)

            self.loss_worker = tf.reduce_mean(
                        -self.surr_worker +
                        config["vf_loss_coeff_worker"] * self.vf_loss_worker -
                        config["entropy_coeff"] * self.entropy_worker)

        with tf.variable_scope("Loss_TOTAL"):

            if not self.ES:
                self.loss_total = tf.reduce_mean(
                            - self.surr_manager
                            + config["vf_loss_coeff_manager"] * self.vf_loss_manager
                            - self.surr_worker
                            + config["vf_loss_coeff_worker"] * self.vf_loss_worker
                            - config["entropy_coeff"] * self.entropy_worker)
            else:
                self.loss_total = tf.reduce_mean(
                    - self.surr_worker
                    + config["vf_loss_coeff_manager"] * self.vf_loss_manager
                    + config["vf_loss_coeff_worker"] * self.vf_loss_worker
                    - config["entropy_coeff"] * self.entropy_worker)

            self.sess = sess

            if config["use_gae"]:
                self.policy_warmup = [
                    self.s, self.g, self.z
                ]

                self.policy_results = [
                            self.sampler, self.curr_logits, self.value_function_manager, self.value_function_worker]


    def compute_manager(self, observation):
        s, g, z  = self.sess.run(
            self.policy_warmup,
            feed_dict={self.observations: [observation]})
        return s, g, z


    def compute_worker(self, g, z, gsum, observation):
        action, logprobs, vfm, vfw = self.sess.run(
                    self.policy_results,
                    feed_dict={self.observations: [observation],
                               self.carried_g: g, self.carried_z: z, self.carried_gsum: gsum})

        return action, {"vf_preds_manager": vfm[0], "vf_preds_worker": vfw, "logprobs": logprobs[0]}

    def loss_total(self):
        return self.loss_total

    def loss_manager(self):
        return self.loss_manager

    def mean_vf_loss_manager(self):
        return self.mean_vf_loss_manager

    def manager_output(self):
        return self.manager_output

    def loss_worker(self):
        return self.loss_worker

    def mean_vf_loss_worker(self):
        return self.mean_vf_loss_worker





