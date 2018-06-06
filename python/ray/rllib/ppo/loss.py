from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ray.rllib.models import ModelCatalog
import tensorflow.contrib.slim as slim
from ray.rllib.models.misc import normc_initializer


class ProximalPolicyLoss(object):
    other_output = ["vf_preds", "logprobs"]
    is_recurrent = False

    def __init__(
            self, observation_space, action_space,
            observations, value_targets, advantages, actions,
            prev_logits, prev_vf_preds, logit_dim,
            kl_coeff, distribution_class, config, sess, registry, ADB):

        print("THIS IS THE NEW PPL")
        self.filer_summaries = []
        self.ADB = ADB
        self.prev_dist = distribution_class(prev_logits)

        # Saved so that we can compute actions given different observations
        self.observations = observations


        self.actions = actions
        action_dim = action_space.shape[0]

        hiddens_policy = config["hiddens_policy"]

        with tf.name_scope("policy_net"):
            i = 1
            last_layer = self.observations
            for size in hiddens_policy:
                label = "policy_net_fc{}".format(i)
                last_layer = slim.fully_connected(
                    last_layer, size,
                    weights_initializer=normc_initializer(1.0),
                    activation_fn=tf.nn.tanh,
                    scope=label)
                i += 1

            label = "policy_net_fc_out"
            self.curr_logits = slim.fully_connected(
                last_layer, logit_dim,
                weights_initializer=normc_initializer(1.0),
                activation_fn=None, scope=label)

            self.curr_dist = distribution_class(self.curr_logits)

        self.sampler = self.curr_dist.sample()


        if config["use_gae"]:

            if self.ADB:
                self.input_value_function = tf.concat([observations, actions], 1)
            else:
                self.input_value_function = self.observations

            hiddens_vf = config["hiddens_vf"]
            with tf.name_scope("value_function_net"):
                i = 1
                last_layer = self.input_value_function
                for size in hiddens_vf:
                    label = "value_function_net_fc{}".format(i)
                    last_layer = slim.fully_connected(
                        last_layer, size,
                        weights_initializer=normc_initializer(1.0),
                        activation_fn=tf.nn.relu,
                        scope=label)
                    i += 1

                label = "value_function_net_fc_out"
                out_ = slim.fully_connected(
                        last_layer, 1,
                        weights_initializer=normc_initializer(1.0),
                        activation_fn=None, scope=label)
                if self.ADB:
                    self.Q_function = tf.reshape(out_, [-1])
                else:
                    self.value_function = tf.reshape(out_, [-1])

        curr_r_matrix = self.curr_dist.r_matrix(actions)
        prev_r_matrix = self.prev_dist.r_matrix(actions)

        curr_logp = self.curr_dist.logp(actions)
        prev_logp = self.prev_dist.logp(actions)
        self.kl = self.prev_dist.kl(self.curr_dist)
        self.mean_kl = tf.reduce_mean(self.kl)
        self.entropy = self.curr_dist.entropy()


        """TRPO"""
        self.mean_entropy = tf.reduce_mean(self.entropy)
        self.vf_loss1 = tf.square(self.value_function - value_targets)
        vf_clipped = prev_vf_preds + tf.clip_by_value(
            self.value_function - prev_vf_preds,
            -config["clip_param"], config["clip_param"])
        self.vf_loss2 = tf.square(vf_clipped - value_targets)
        self.vf_loss = tf.minimum(self.vf_loss1, self.vf_loss2)
        self.mean_vf_loss = tf.reduce_mean(self.vf_loss)

        self.ratio = tf.exp(curr_logp - prev_logp)
        self.surr = self.ratio * advantages

        self.mean_policy_loss = tf.reduce_mean(- self.surr + kl_coeff * self.kl)
        self.loss = tf.reduce_mean(
            -self.surr + kl_coeff * self.kl +
            config["vf_loss_coeff"] * self.vf_loss)


        """
        # Make loss functions.
        if ADB:
            self.ratio = curr_r_matrix / prev_r_matrix
            self.surr2 = tf.clip_by_value(self.ratio, (1 - config["clip_param"]) ** (1 / action_dim),
                                           (1 + config["clip_param"]) ** (1 / action_dim)) * advantages
        else:
            self.ratio = tf.exp(curr_logp - prev_logp)

            self.surr2 = tf.clip_by_value(self.ratio, 1 - config["clip_param"],
                                           1 + config["clip_param"]) * advantages


        self.mean_entropy = tf.reduce_mean(self.entropy)

        self.surr1 = self.ratio * advantages


        if ADB:
            self.surr = tf.reduce_sum(tf.minimum(self.surr1, self.surr2), reduction_indices=[1])
        else:
            self.surr = tf.minimum(self.surr1, self.surr2)

        self.mean_policy_loss = - tf.reduce_mean(self.surr)

        if config["use_gae"]:
            # We use a huber loss here to be more robust against outliers,
            # which seem to occur when the rollouts get longer (the variance
            # scales superlinearly with the length of the rollout)
            if ADB:
                self.vf_loss1 = tf.square(self.Q_function - value_targets)
                vf_clipped = prev_vf_preds + tf.clip_by_value(
                    self.Q_function - prev_vf_preds,
                    -config["clip_param"], config["clip_param"])
            else:
                self.vf_loss1 = tf.square(self.value_function - value_targets)
                vf_clipped = prev_vf_preds + tf.clip_by_value(
                    self.value_function - prev_vf_preds,
                    -config["clip_param"], config["clip_param"])

            self.vf_loss2 = tf.square(vf_clipped - value_targets)
            self.vf_loss = tf.minimum(self.vf_loss1, self.vf_loss2)
            self.mean_vf_loss = tf.reduce_mean(self.vf_loss)

            self.loss = tf.reduce_mean(
                    -self.surr + kl_coeff * self.kl +
                    config["vf_loss_coeff"] * self.vf_loss -
                    config["entropy_coeff"] * self.entropy)
                    
        """


        else:
            self.mean_vf_loss = tf.constant(0.0)
            self.loss = tf.reduce_mean(0)

        self.sess = sess

        if config["use_gae"]:
            if ADB:
                self.policy_results = [
                    self.sampler, self.curr_logits]
            else:
                self.policy_results = [
                    self.sampler, self.curr_logits, self.value_function]
        else:
            self.policy_results = [
                self.sampler, self.curr_logits, tf.constant("NA")]


    def compute(self, observation):
        if self.ADB:
            action, logprobs = self.sess.run(
                    self.policy_results,
                    feed_dict={self.observations: [observation]})

            vf = self.sess.run(
                    self.Q_function,
                    feed_dict={self.observations: [observation], self.actions: action})

            return action[0], {"vf_preds": vf[0], "logprobs": logprobs[0]}

        else:
            action, logprobs, vf = self.sess.run(
                        self.policy_results,
                        feed_dict={self.observations: [observation]})
            return action[0], {"vf_preds": vf[0], "logprobs": logprobs[0]}


    def compute_Q_fuctions(self, observations, actions):
        Q_functions = []
        import numpy as np
        actions = np.array(actions)
        means = np.mean(actions, axis=0)
        for j in range(actions.shape[1]):
            actions_j = np.copy(actions)
            actions_j[:, j] = means[j]
            Q_functions.append(self.sess.run(
                self.Q_function,
                feed_dict={self.observations: observations, self.actions: actions_j}))
        return Q_functions

    def loss(self):
        return self.loss

    def mean_vf_loss(self):
        return self.mean_vf_loss
