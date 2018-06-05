from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ray.rllib.models import ModelCatalog
from ray.rllib.models.singlestepLSTM import SingleStepLSTM
import numpy as np



class FeudalLoss(object):


    other_output = ["vf_preds_manager", "vf_preds_worker", "logprobs"]
    other_output_ES = ["vf_preds_worker", "logprobs"]
    is_recurrent = False

    def __init__(
            self, gsum, observation_space, action_space,
            observations, value_targets_worker, advantages_worker, actions,
            distribution_class_obs, config, sess, registry, ES,
            diff=None, value_targets_manager=None, advantages_manager=None):

        self.ES = ES
        self.action_dim= action_space.n
        self.actions = actions
        # Saved so that we can compute actions given different observations
        self.observations = tf.cast(observations, tf.float32)
        self.g_sum = tf.stop_gradient(gsum)
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
                                         units=256, \
                                         activation=tf.nn.relu)



        with tf.variable_scope("Manager"):
            self.s = tf.layers.dense(inputs=self.z, \
                                 units=config["g_dim"], \
                                 activation=tf.nn.elu)

            x = tf.expand_dims(self.s, [0])

            with tf.variable_scope("LSTM"):
                self.manager_lstm = SingleStepLSTM(size=config["g_dim"], dilatation_rate=config["dilatation_rate"])
                g_hat = self.manager_lstm.compute_step(x, step_size=tf.shape(self.observations)[:1])

            g_hat = tf.reshape(g_hat, shape=(-1, config["g_dim"]))
            self.g = tf.nn.l2_normalize(g_hat, dim=1)

            hidden_manager = tf.layers.dense(inputs=g_hat, \
                                     units=config["vf_hidden_size"], \
                                     activation=tf.nn.elu)

            weights_VF_manager = tf.get_variable("weights_VF_manager", (config["vf_hidden_size"], 1))
            self.value_function_manager = tf.matmul(hidden_manager, weights_VF_manager)

            if self.ES:
                self.manager_logits = self.g

            else:
                self.manager_logits = distribution_class_obs(self.g, config["kappa"], observation_space.shape[0])
                vf_config = config["model"].copy()
                vf_config["free_log_std"] = False
                with tf.variable_scope("value_function_manager"):
                    self.value_function_manager = ModelCatalog.get_model(
                        registry, observations, 1, vf_config).outputs
                self.value_function_manager = tf.reshape(self.value_function_manager, [-1])
                self.diff = diff
                self.diff = tf.nn.l2_normalize(self.diff, dim=1)
                self.logp_manager = self.manager_logits.logp(self.diff)

                self.surr_manager = self.logp_manager * advantages_manager
                self.mean_surr_manager = tf.reduce_mean(self.surr_manager)

                self.vf_loss_manager = tf.square(self.value_function_manager - value_targets_manager)
                self.mean_vf_loss_manager = tf.reduce_mean(self.vf_loss_manager)

                self.loss_manager = tf.reduce_mean(
                        -self.surr_manager +
                        config["vf_loss_coeff_manager"] * self.vf_loss_manager)

        with tf.variable_scope("Worker"):


            with tf.variable_scope("LSTM"):

                new_z = tf.stop_gradient(self.z)
                dimension_lstm_worker = self.action_dim * config["k"]
                lstm_cell_worker = tf.nn.rnn_cell.BasicLSTMCell(dimension_lstm_worker)
                initial_state = lstm_cell_worker.zero_state(1, dtype=tf.float32)
                outputs_worker, state = tf.nn.dynamic_rnn(lstm_cell_worker, tf.expand_dims(new_z, [0]),
                                                   initial_state=initial_state,
                                                   dtype=tf.float32)


                outputs_worker = tf.squeeze(outputs_worker, axis=0)

            hidden_VF_worker = tf.layers.dense(inputs=outputs_worker, \
                                     units=config["vf_hidden_size"], \
                                     activation=tf.nn.elu)

            weights_VF_worker = tf.get_variable("weights", (config["vf_hidden_size"], self.action_dim))
            self.value_function_worker = tf.matmul(hidden_VF_worker, weights_VF_worker)

            print("self.value_function_worker")
            print(self.value_function_worker)

            U = tf.reshape(outputs_worker, [-1, self.action_dim, config["k"]])

            # Calculate w

            phi = tf.get_variable("phi", (config["g_dim"], config['k']))
            w = tf.matmul(self.g_sum, phi)
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

            print("self.sample")
            print(self.sampler)



            self.entropy_worker =-tf.reduce_sum(self.pi * self.log_pi)
            self.mean_entropy_worker = tf.reduce_mean(self.entropy_worker)

            self.vf_loss_worker = tf.square(self.value_function_worker - value_targets_worker)
            self.mean_vf_loss_worker = tf.reduce_mean(self.vf_loss_worker)

            self.surr_worker = tf.reduce_sum(self.log_pi * advantages_worker, [1])
            self.mean_policy_loss_worker = -tf.reduce_mean(self.surr_worker)

            self.loss_worker = tf.reduce_mean(
                        -self.surr_worker +
                        config["vf_loss_coeff_worker"] * self.vf_loss_worker -
                        config["entropy_coeff"] * self.entropy_worker)

            self.sess = sess

            if config["use_gae"]:
                self.policy_warmup = [
                    self.s, self.g
                ]

                self.policy_results = [
                            self.sampler, self.curr_logits, self.value_function_manager, self.value_function_worker]


    def compute_manager(self, observation):
        s, g  = self.sess.run(
            self.policy_warmup,
            feed_dict={self.observations: [observation]})
        return s, g

    def compute_worker(self, gsum, observation):
        action, logprobs, vfm, vfw = self.sess.run(
                    self.policy_results,
                    feed_dict={self.observations: [observation], self.g_sum: gsum})

        return action, {"vf_preds_manager": vfm[0], "vf_preds_worker": vfw, "logprobs": logprobs[0]}


    def loss_manager(self):
        return self.loss_manager

    def mean_vf_loss_manager(self):
        return self.mean_vf_loss_manager

    def manager_logits(self):
        return self.manager_logits

    def loss_worker(self):
        return self.loss_worker

    def mean_vf_loss_worker(self):
        return self.mean_vf_loss_worker





