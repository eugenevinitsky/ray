from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from ray.rllib.models import ModelCatalog


class ProximalPolicyLoss(object):

    other_output = ["vf_preds", "logprobs"]
    is_recurrent = False

    def __init__(
            self, observation_space, action_space,
            observations, value_targets, advantages, actions,
            prev_logits, prev_vf_preds, logit_dim,
            kl_coeff, distribution_class, config, sess, registry):
        self.prev_dist = distribution_class(prev_logits)

        # Saved so that we can compute actions given different observations
        self.observations = observations

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
                self.value_function = ModelCatalog.get_model(
                    registry, observations, 1, vf_config).outputs
            self.value_function = tf.reshape(self.value_function, [-1])

        # Make loss functions.
        self.ratio = tf.exp(self.curr_dist.logp(actions) -
                            self.prev_dist.logp(actions))
        self.kl = self.prev_dist.kl(self.curr_dist)
        self.mean_kl = tf.reduce_mean(self.kl)
        self.entropy = self.curr_dist.entropy()
        self.mean_entropy = tf.reduce_mean(self.entropy)
        self.surr1 = self.ratio * advantages
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
            self.loss = tf.reduce_mean(
                -self.surr + kl_coeff * self.kl +
                config["vf_loss_coeff"] * self.vf_loss -
                config["entropy_coeff"] * self.entropy)
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

    def compute(self, observation):
        import ipdb; ipdb.set_trace()
        action, logprobs, vf = self.sess.run(
            self.policy_results,
            feed_dict={self.observations: [observation]})
        return action[0], {"vf_preds": vf[0], "logprobs": logprobs[0]}

    def loss(self):
        return self.loss


class mProximalPolicyLoss:
    other_output = ["vf_preds", "logprobs"]
    is_recurrent = False

    def __init__(
            self, env,
            observations, value_targets, advantages, actions,
            prev_logits, prev_vf_preds, logit_dim,
            kl_coeff, distribution_class, config, sess):
        assert (isinstance(env.action_space, gym.spaces.Discrete) or
                isinstance(env.action_space, gym.spaces.Box))

        # store a copy of the observations for the feed dict
        self.observations = observations
        split_observations = env.split_input_tensor(observations)
        act_list = env.split_output_tensor(actions)
        prev_logits = env.split_output_tensor(prev_logits)
        split_advantages = env.split_along_agents(advantages)
        # FIXME this is not getting split right and it should be logit_dim

        split_logit_dim = env.split_output_number(logit_dim)

        self.prev_dist = []
        self.curr_logits = []
        self.curr_dist = []
        self.sampler = []
        self.ratio = []
        self.kl = []
        self.mean_kl = []
        self.entropy = []
        self.mean_entropy = []
        self.surr1 = []
        self.surr2 = []
        self.surr = []
        self.mean_policy_loss = []
        # TODO is there some way to remove this list and just make it over tensors?
        for i in range(len(act_list)):
            config["model_num"] = str(i)
            self.prev_dist.append(distribution_class(prev_logits[i]))
            self.curr_logits.append(ModelCatalog.get_model(
                split_observations[i], int(split_logit_dim[i]), config).outputs)
            self.curr_dist.append(distribution_class(self.curr_logits[i]))
            self.sampler.append(self.curr_dist[i].sample())

            # Make loss functions.
            self.ratio.append(tf.exp(self.curr_dist[i].logp(act_list[i]) -
                                self.prev_dist[i].logp(act_list[i])))
            self.kl.append(self.prev_dist[i].kl(self.curr_dist[i]))
            self.mean_kl.append(tf.reduce_mean(self.kl[i]))
            self.entropy.append(self.curr_dist[i].entropy())
            self.mean_entropy.append(tf.reduce_mean(self.entropy))
            self.surr1.append(self.ratio[i]* split_advantages[i])
            self.surr2.append(tf.clip_by_value(self.ratio[i], 1 - config["clip_param"],
                                          1 + config["clip_param"]) * split_advantages[i])
            self.surr.append(tf.minimum(self.surr1[i], self.surr2[i]))
            self.mean_policy_loss.append(tf.reduce_mean(-self.surr[i]))


        # TODO implement this properly for multiagents
        if config["use_gae"]:
            vf_config = config["model"].copy()
            # Do not split the last layer of the value function into
            # mean parameters and standard deviation parameters and
            # do not make the standard deviations free variables.
            vf_config["free_log_std"] = False
            with tf.variable_scope("value_function"):
                self.value_function = ModelCatalog.get_model(
                    observations, 1, vf_config).outputs
            self.value_function = tf.reshape(self.value_function, [-1])


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
            self.loss = tf.reduce_mean(
                -self.surr + kl_coeff * self.kl +
                config["vf_loss_coeff"] * self.vf_loss -
                config["entropy_coeff"] * self.entropy)
        else:
            # FIXME what is this for?
            self.mean_vf_loss = tf.constant(0.0)

            # FIXME this probably doesn't work if the tensors have different shape
            self.loss = tf.reduce_mean(
                -tf.add_n(self.surr) +
                kl_coeff * tf.add_n(self.kl) -
                config["entropy_coeff"] * tf.add_n(self.entropy))

        self.sess = sess

        if config["use_gae"]:
            self.policy_results = [
                self.sampler, self.curr_logits, self.value_function]
        else:
            # FIXME listify this
            # policy results is a list
            self.policy_results = [
                self.sampler, self.curr_logits, tf.constant("NA")]

    def compute(self, observation):

        # import ipdb; ipdb.set_trace()
        # FIXME(ev) the shape of the logprobs (and actions sort of) are causing issues later
        action, logprobs, vf = self.sess.run(
            self.policy_results,
            feed_dict={self.observations: [observation]})
        # import ipdb; ipdb.set_trace()
        return action, {"vf_preds": vf, "logprobs": logprobs}

    def loss(self):
        return self.loss