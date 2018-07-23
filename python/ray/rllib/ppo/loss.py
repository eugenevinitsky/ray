from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from ray.rllib.models import ModelCatalog


class ProximalPolicyGraph(object):

    other_output = ["vf_preds", "logprobs"]
    is_recurrent = False

    def __init__(
            self, observation_space, action_space,
            observations, value_targets, advantages, actions,
            prev_logits, prev_vf_preds, logit_dim,
            kl_coeff, distribution_class, config, sess, registry, ADB=False):
        self.prev_dist = distribution_class(prev_logits)

        # Saved so that we can compute actions given different observations
        self.observations = observations
        self.actions = actions
        self.curr_logits = ModelCatalog.get_model(
            registry, observations, logit_dim, config["model"]).outputs
        self.curr_dist = distribution_class(self.curr_logits)
        self.sampler = self.curr_dist.sample()

        if config["use_gae"]:
            vf_config = config["model"].copy()
            # Do not split the last layer of the value function into
            # mean parameters and standard deviation parameters and
            # do not make the standard deviations free variables.
            vf_config["free_log_std"] = False
            with tf.variable_scope("value_function"):
                if ADB:
                    zero_action = tf.zeros(tf.shape(self.actions))
                    input_vf = tf.concat([self.observations, zero_action], 1)
                    #input_vf = tf.concat([self.observations, self.actions], 1)
                else:
                    input_vf = self.observations
                self.value_function = ModelCatalog.get_model(
                    registry, input_vf, 1, vf_config).outputs
            self.value_function = tf.reshape(self.value_function, [-1])

        # Make loss functions.
        if ADB:
            self.ratio = tf.div(self.curr_dist.r_matrix(actions),
                            self.prev_dist.r_matrix(actions))
        else:
            self.ratio = tf.exp(self.curr_dist.logp(actions) -
                                self.prev_dist.logp(actions))
        self.kl = self.prev_dist.kl(self.curr_dist)
        self.mean_kl = tf.reduce_mean(self.kl)
        self.entropy = self.curr_dist.entropy()
        self.mean_entropy = tf.reduce_mean(self.entropy)
        self.surr1 = self.ratio * advantages
        if ADB:

            action_dim = action_space.shape[0]
            self.surr2 = tf.clip_by_value(self.ratio, (1 - config["clip_param"])**(1/action_dim),
                                          (1 + config["clip_param"])**(1/action_dim)) * advantages
            """
            self.surr2 = tf.clip_by_value(self.ratio, 1 - config["clip_param"],
                                          1 + config["clip_param"]) * advantages
            """
            self.surr = tf.reduce_sum(tf.minimum(self.surr1, self.surr2), reduction_indices=[1])

        else:
            self.surr2 = tf.clip_by_value(self.ratio, 1 - config["clip_param"],
                                          1 + config["clip_param"]) * advantages
            self.surr = tf.minimum(self.surr1, self.surr2)
        self.mean_policy_loss = tf.reduce_mean(-self.surr)

        if config["use_gae"]:
            # We use a huber loss here to be more robust against outliers,
            # which seem to occur when the rollouts get longer (the variance
            # scales superlinearly with the length of the rollout)
            self.vf_loss1 = tf.square(self.value_function - value_targets)
            vf_clipped = prev_vf_preds + tf.clip_by_value(
                self.value_function - prev_vf_preds,
                -config["clip_param"], config["clip_param"])
            self.vf_loss2 = tf.square(vf_clipped - value_targets)
            self.vf_loss = tf.minimum(self.vf_loss1, self.vf_loss2)
            self.mean_vf_loss = tf.reduce_mean(self.vf_loss)
            self.loss = tf.reduce_mean(-self.surr)
            self.loss_vf = tf.reduce_mean(self.vf_loss)

        else:
            self.mean_vf_loss = tf.constant(0.0)
            self.loss = tf.reduce_mean(
                -self.surr +
                kl_coeff * self.kl -
                config["entropy_coeff"] * self.entropy)

        self.sess = sess

        if config["use_gae"]:
            self.policy_results = [
                self.sampler, self.curr_logits, self.value_function]
        else:
            self.policy_results = [
                self.sampler, self.curr_logits, tf.constant("NA")]

    def compute_single_action(self, observation, features, is_training=False):
        action, logprobs = self.sess.run(
            [self.sampler, self.curr_logits],
            feed_dict={self.observations: [observation]})
        vf = self.sess.run(
            self.value_function,
            feed_dict={self.observations: [observation], self.actions: action})

        return action[0], [], {"vf_preds": vf[0], "logprobs": logprobs[0]}

    def compute_ad_baseline(self, observations, actions):
        Q_functions = []
        actions = np.array(actions)
        means = np.mean(actions, axis=0)
        for j in range(actions.shape[1]):
            actions_j = np.copy(actions)
            actions_j[:, j] = means[j]
            Q_functions.append(self.sess.run(
                self.value_function,
                feed_dict={self.observations: observations, self.actions: actions_j}))
        return Q_functions

    def get_initial_state(self):
        return []

    def loss(self):
        return self.loss

    def loss_vf(self):
        return self.loss_vf