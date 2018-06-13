from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ray.rllib.models import ModelCatalog
import tensorflow.contrib.slim as slim
from ray.rllib.models.misc import normc_initializer


class ProximalPolicyLoss(object):
    other_output = []
    is_recurrent = False

    def __init__(
            self, observation_space, action_space, observations,
            value_targets_manager, value_targets_worker, advantages_manager, advantages_worker,
            actions, prev_logits, prev_manager_vf_preds, prev_worker_vf_preds, s_diff, g_sum, z_carried,
            logit_dim, kl_coeff, distribution_class, config, sess, registry):

        self.prev_dist = distribution_class(prev_logits)

        # Saved so that we can compute actions given different observations
        self.observations = tf.cast(observations, tf.float32)
        self.s_diff = tf.stop_gradient(s_diff)
        self.g_sum = tf.stop_gradient(g_sum)
        self.z_carried = tf.stop_gradient(z_carried)
        self.actions = actions

        vf_critic_config = config["model_manager_critic"].copy()
        with tf.variable_scope("value_function_manager"):
            self.value_function_manager = ModelCatalog.get_model(
                    registry, self.observations , 1, vf_critic_config).outputs
            self.value_function_manager = tf.reshape(self.value_function_manager, [-1])
            print("self.value_function_manager")
            print(self.value_function_manager)

        with tf.variable_scope("Manager"):
            vf_manager_config = config["model_manager"].copy()
            self.z = ModelCatalog.get_model(
                registry, observations, config["z_dimension"], vf_manager_config).outputs
            self.z = tf.reshape(self.z, [-1, config["z_dimension"]])

            self.s = slim.fully_connected(
                self.z, config["g_dim"],
                weights_initializer=normc_initializer(1.0),
                activation_fn=tf.nn.tanh,
                scope="s")

            #x = tf.expand_dims(self.s, [0])

            with tf.variable_scope("Goal_Designer"):
                goal_designer_manager = config["model_goal_designer_manager"].copy()
                model_manager = ModelCatalog.get_model(
                    registry, self.s , config["g_dim"], goal_designer_manager).outputs

                g_hat = tf.reshape(model_manager, [-1, config["g_dim"]])


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

            with tf.variable_scope("Goal_Embedding"):
                model_goal_embedding_worker = config["model_goal_embedding_worker"].copy()
                model_worker= ModelCatalog.get_model(
                    registry, self.z_carried, logit_dim * config['k'], model_goal_embedding_worker)

                U = tf.reshape(model_worker.outputs, [-1, logit_dim, config['k']])

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

            self.ratio = tf.exp(self.curr_dist.logp(actions) -
                                self.prev_dist.logp(actions))
            self.surr1 = self.ratio * advantages_worker
            self.surr2 = tf.clip_by_value(self.ratio, 1 - config["clip_param"],
                                              1 + config["clip_param"]) * advantages_worker
            self.surr_worker = - tf.minimum(self.surr1, self.surr2)

            self.mean_worker_policy_loss = tf.reduce_mean(self.surr_worker)

        with tf.variable_scope("TOTAL_LOSS"):
            self.kl = self.prev_dist.kl(self.curr_dist)
            self.mean_kl = tf.reduce_mean(self.kl)
            self.entropy = self.curr_dist.entropy()
            self.mean_entropy = tf.reduce_mean(self.entropy)

            self.loss = tf.reduce_mean(
                self.manager_surr +
                config["manager_vf_loss_coeff"] * self.manager_vf_loss +
                self.surr_worker +
                config["worker_vf_loss_coeff"] * self.worker_vf_loss +
                kl_coeff * self.kl -
                config["entropy_coeff"] * self.entropy
            )

        self.sess = sess


    def compute_manager(self, observation):
        z, s, g, vfm = self.sess.run(
            [self.z, self.s, self.g, self.value_function_manager],
            feed_dict={self.observations: [observation]})
        return z[0], s[0], g[0], vfm[0]

    def compute_worker(self, z_carried, g_sum):
        action, logprobs, vfw = self.sess.run(
            [self.sampler, self.curr_logits, self.worker_value_function],
            feed_dict={self.z_carried: [z_carried], self.g_sum: [g_sum]})
        return action[0], logprobs[0], vfw[0]

    def loss(self):
        return self.loss