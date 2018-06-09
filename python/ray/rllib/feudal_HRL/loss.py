from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ray.rllib.models import ModelCatalog
import tensorflow.contrib.slim as slim
from ray.rllib.models.misc import normc_initializer


class ProximalPolicyLoss(object):


    is_recurrent = False

    def __init__(
            self, observation_space, action_space, observations,
            value_targets_manager, value_targets_worker, advantages_manager, advantages_worker,
            actions, prev_logits, prev_manager_vf_preds, prev_worker_vf_preds, s_diff, g_sum, z_carried,
            c_in_manager, h_in_manager, c_in_worker, h_in_worker, logit_dim,
            kl_coeff, distribution_class, config, sess, registry):

        self.prev_dist = distribution_class(prev_logits)

        # Saved so that we can compute actions given different observations
        self.observations = observations
        self.s_diff = s_diff
        self.g_sum = g_sum
        self.z_carried = z_carried
        self.actions = actions
        self.c_in_manager_input = c_in_manager
        self.h_in_manager_input = h_in_manager
        self.c_in_worker_input = c_in_worker
        self.h_in_worker_input = h_in_worker

        self.action_dim = self.actions.shape[0]

        vf_critic_config = config["model_manager_critic"].copy()
        with tf.variable_scope("value_function_manager"):
            self.value_function_manager = ModelCatalog.get_model(
                    registry, observations, 1, vf_critic_config).outputs
            self.value_function_manager = tf.reshape(self.value_function_manager, [-1])

        with tf.variable_scope("Manager"):
            vf_manager_config = config["model_manager"].copy()
            self.z = ModelCatalog.get_model(
                registry, observations, config["z_dimension"], vf_manager_config).outputs


            self.s = slim.fully_connected(
                self.z, config["g_dim"],
                weights_initializer=normc_initializer(1.0),
                activation_fn=tf.nn.elu,
                scope="s")

            with tf.variable_scope("LSTM"):
                lstm_manager_config = config["model_LTSM_manager"].copy()
                inputs_lstm_manager = [self.s, self.c_in_manager_input, self.h_in_manager_input]
                model_manager = ModelCatalog.get_model(
                    registry, inputs_lstm_manager , config["g_dim"], lstm_manager_config)

                g_hat = model_manager.outputs
                self.c_in_manager, self.h_in_manager = model_manager.last_layer
            self.g = tf.nn.l2_normalize(g_hat, dim=1)

        with tf.variable_scope("Manager_LOSS"):

            self.manager_vf_loss1 = tf.square(self.value_function_manager - value_targets_manager)
            manager_vf_clipped = prev_manager_vf_preds + tf.clip_by_value(
                self.value_function_manager - prev_manager_vf_preds,
                -config["clip_param"], config["clip_param"])
            self.manager_vf_loss2 = tf.square(manager_vf_clipped - value_targets_manager)
            self.manager_vf_loss = tf.minimum(self.manager_vf_loss1, self.manager_vf_loss2)
            self.mean_manager_vf_loss = tf.reduce_mean(self.manager_vf_loss)

            dot = tf.reduce_sum(tf.multiply(self.s_diff, self.g), axis=1)
            gcut = tf.stop_gradient(self.g)
            mag = tf.norm(self.s_diff, axis=1) * tf.norm(gcut, axis=1) + .0001
            dcos = dot / mag
            self.manager_surr = - dcos * advantages_manager
            self.mean_manager_policy_loss = tf.reduce_mean(self.manager_surr)

        with tf.variable_scope("worker_critic"):
            worker_vf_config = config["model_worker_critic"].copy()
            self.worker_value_function = ModelCatalog.get_model(
                    registry, self.z_carried, 1, worker_vf_config).outputs
            self.worker_value_function = tf.reshape(self.worker_value_function, [-1])

        with tf.variable_scope("Worker"):
            phi = tf.get_variable("phi", (config["g_dim"], config['k']))
            w = tf.matmul(self.g_sum, phi)
            w = tf.expand_dims(w, [2])

            with tf.variable_scope("LSTM"):
                lstm_worker_config = config["model_LTSM_worker"].copy()
                inputs_lstm_worker = [self.z_carried, self.c_in_worker_input, self.h_in_worker_input]
                model_worker= ModelCatalog.get_model(
                    registry, inputs_lstm_worker, logit_dim * config['k'], lstm_worker_config)

                U = tf.reshape(model_worker.outputs, [-1, logit_dim, config['k']])
                self.c_in_worker, self.h_in_worker = model_worker.last_layer

            self.curr_logits = tf.reshape(tf.matmul(U, w),[-1, logit_dim])
            self.curr_dist = distribution_class(self.curr_logits)
            self.sampler = self.curr_dist.sample()

        with tf.variable_scope("Worker_LOSS"):
            self.worker_vf_loss1 = tf.square(self.worker_value_function - value_targets_worker)
            worker_vf_clipped = prev_worker_vf_preds + tf.clip_by_value(
                self.worker_value_function - prev_worker_vf_preds,
                -config["clip_param"], config["clip_param"])
            self.worker_vf_loss2 = tf.square(worker_vf_clipped - value_targets_worker)
            self.worker_vf_loss = tf.minimum(self.worker_vf_loss1, self.worker_vf_loss2)
            self.mean_worker_vf_loss = tf.reduce_mean(self.worker_vf_loss)

            self.surr_worker = - self.curr_dist.logp(self.actions) * advantages_worker
            self.mean_worker_policy_loss = tf.reduce_mean(self.surr_worker)

        with tf.variable_scope("TOTAL_LOSS"):
            self.kl = self.prev_dist.kl(self.curr_dist)
            self.mean_kl = tf.reduce_mean(self.kl)
            self.entropy = self.curr_dist.entropy()
            self.mean_entropy = tf.reduce_mean(self.entropy)

            self.loss = tf.reduce_mean(
                self.surr_worker +
                config["manager_vf_loss_coeff"] * self.manager_vf_loss +
                self.surr_worker +
                config["worker_vf_loss_coeff"] * self.worker_vf_loss +
                kl_coeff * self.kl -
                config["entropy_coeff"] * self.entropy
            )

        self.sess = sess


    def compute_manager(self, observation, c_in_manager_input, h_in_manager_input):
        z, s, g, c_out_manager, h_out_manager, vfm = self.sess.run(
            [self.z, self.s, self.g, self.c_in_manager, self.h_in_manager, self.value_function_manager],
            feed_dict={self.observations: [observation],
                       self.c_in_manager_input: c_in_manager_input, self.h_in_worker_input: h_in_manager_input})
        return z, s, g, c_out_manager, h_out_manager, vfm[0]

    def compute_worker(self, z, g_sum, c_in_worker_input, h_in_worker_input):
        action, c_out_worker, h_out_worker, logprobs, vfw = self.sess.run(
            [self.sampler, self.c_in_worker, self.h_in_worker, self.curr_logits, self.worker_value_function],
            feed_dict={self.z_carried: z, self.g_sum: g_sum,
                       self.c_in_worker_input: c_in_worker_input, self.h_in_worker_input: h_in_worker_input})
        return action[0], c_out_worker, h_out_worker, logprobs[0], vfw[0]

    def loss(self):
        return self.loss