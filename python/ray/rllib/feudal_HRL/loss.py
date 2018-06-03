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
            observations, value_targets_worker, advantages_worker, actions, prev_logits, prev_vf_preds_worker,
            logit_dim, kl_coeff, distribution_class, distribution_class_obs, config, sess, registry, ADB, ES,
            diff=None, prev_vf_preds_manager=None, value_targets_manager=None, advantages_manager=None):

        self.ES = ES
        self.ADB = ADB
        action_dim = action_space.shape[0]
        self.actions = actions
        self.prev_dist = distribution_class(prev_logits)
        # Saved so that we can compute actions given different observations
        self.observations = observations
        self.g_sum = tf.stop_gradient(gsum)
        self.diff = diff


        if config["activate_filter"]:
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
        else:
            with tf.variable_scope("z"):
                self.z = tf.layers.dense(inputs=self.observations, \
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

                self.vf_loss1_manager = tf.square(self.value_function_manager - value_targets_manager)
                vf_clipped_manager = prev_vf_preds_manager + tf.clip_by_value(
                    self.value_function_manager - prev_vf_preds_manager,
                    -config["clip_param"], config["clip_param"])
                self.vf_loss2_manager = tf.square(vf_clipped_manager - value_targets_manager)
                self.vf_loss_manager = tf.minimum(self.vf_loss1_manager, self.vf_loss2_manager)
                self.mean_vf_loss_manager = tf.reduce_mean(self.vf_loss_manager)

                self.loss_manager = tf.reduce_mean(
                        -self.surr_manager +
                        config["vf_loss_coeff_manager"] * self.vf_loss_manager)

        with tf.variable_scope("Worker"):

            if self.ADB:
                self.input_Q_value = tf.concat([observations, actions], 1)

            vf_config = config["model"].copy()
            # Do not split the last layer of the value function into
            # mean parameters and standard deviation parameters and
            # do not make the standard deviations free variables.
            vf_config["free_log_std"] = False

            if self.ADB:
                with tf.variable_scope("Q_function_worker"):
                    self.Q_function_worker = ModelCatalog.get_model(
                        registry, observations, 1, vf_config).outputs
                self.Q_function_worker = tf.reshape(self.Q_function_worker, [-1])
            else:
                with tf.variable_scope("Value_function_worker"):
                    self.value_function_worker = ModelCatalog.get_model(
                    registry, observations, 1, vf_config).outputs
                self.value_function_worker = tf.reshape(self.value_function_worker, [-1])

            num_acts = action_space.shape[0]

            with tf.variable_scope("LSTM"):

                new_z = tf.stop_gradient(self.z)
                dimension_lstm_worker = 2 * num_acts * config["k"]
                lstm_cell_worker = tf.nn.rnn_cell.BasicLSTMCell(dimension_lstm_worker)
                initial_state = lstm_cell_worker.zero_state(1, dtype=tf.float32)
                outputs_worker, state = tf.nn.dynamic_rnn(lstm_cell_worker, tf.expand_dims(new_z, [0]),
                                                   initial_state=initial_state,
                                                   dtype=tf.float32)
                outputs_worker = tf.squeeze(outputs_worker)
                U = tf.reshape(outputs_worker, [-1, 2 * num_acts, config["k"]])

            # Calculate w

            phi = tf.get_variable("phi", (config["g_dim"], config['k']))
            w = tf.matmul(self.g_sum, phi)
            w = tf.expand_dims(w, [2])
            # Calculate policy and sample
            self.curr_logits = tf.reshape(tf.matmul(U, w), [-1, 2 * num_acts])
            self.curr_dist = distribution_class(self.curr_logits)
            self.sampler = self.curr_dist.sample()

            curr_r_matrix = self.curr_dist.r_matrix(actions)
            prev_r_matrix = self.prev_dist.r_matrix(actions)
            curr_logp = self.curr_dist.logp(actions)
            prev_logp = self.prev_dist.logp(actions)
            self.kl = self.prev_dist.kl(self.curr_dist)
            self.entropy_worker = self.curr_dist.entropy()


            if self.ADB:
                self.ratio_worker = curr_r_matrix / prev_r_matrix

                self.surr2_worker = tf.clip_by_value(self.ratio_worker, (1 - config["clip_param"]) ** (1 / action_dim),
                                                      (1 + config["clip_param"]) ** (
                                                                  1 / action_dim)) * advantages_worker

            else:
                self.ratio_worker = tf.exp(curr_logp - prev_logp)

                self.surr2_worker = tf.clip_by_value(self.ratio_worker, 1 - config["clip_param"],
                                                      1 + config["clip_param"]) * advantages_worker


            self.mean_kl = tf.reduce_mean(self.kl)
            self.mean_entropy_worker = tf.reduce_mean(self.entropy_worker)
            self.surr1_worker = self.ratio_worker * advantages_worker

            if self.ADB:
                self.surr_worker = tf.reduce_sum(tf.minimum(self.surr1_worker, self.surr2_worker), reduction_indices=[1])
            else:
                self.surr_worker = tf.minimum(self.surr1_worker, self.surr2_worker)

            self.mean_policy_loss_worker = tf.reduce_mean(-self.surr_worker)

            self.mean_entropy_worker = tf.reduce_mean(self.entropy_worker)

            if config["use_gae"]:
                # We use a huber loss here to be more robust against outliers,
                # which seem to occur when the rollouts get longer (the variance
                # scales superlinearly with the length of the rollout)
                if self.ADB:
                    self.vf_loss1_worker= tf.square(self.Q_function_worker - value_targets_worker)
                    vf_clipped_worker = prev_vf_preds_worker + tf.clip_by_value(
                        self.Q_function_worker - prev_vf_preds_worker,
                        -config["clip_param"], config["clip_param"])
                else:
                    self.vf_loss1_worker = tf.square(self.value_function_worker - value_targets_worker)
                    vf_clipped_worker = prev_vf_preds_worker + tf.clip_by_value(
                        self.value_function_worker - prev_vf_preds_worker,
                        -config["clip_param"], config["clip_param"])

                self.vf_loss2_worker = tf.square(vf_clipped_worker - value_targets_worker)
                self.vf_loss_worker = tf.minimum(self.vf_loss1_worker, self.vf_loss2_worker)
                self.mean_vf_loss_worker = tf.reduce_mean(self.vf_loss_worker)

                self.loss_worker = tf.reduce_mean(
                        -self.surr_worker + kl_coeff * self.kl  +
                        config["vf_loss_coeff_worker"] * self.vf_loss_worker -
                        config["entropy_coeff"] * self.entropy_worker)

            else:
                self.mean_vf_loss_worker = tf.constant(0.0)
                self.loss_worker = tf.reduce_mean(0)

            self.sess = sess


            if config["use_gae"]:
                self.policy_warmup = [
                    self.s, self.g
                ]

                if self.ADB:
                    if self.ES:
                        self.policy_results = [
                            self.sampler, self.curr_logits]
                    else:
                        self.policy_results = [
                            self.sampler, self.curr_logits, self.value_function_manager]
                else:
                    if self.ES:
                        self.policy_results = [
                            self.sampler, self.curr_logits, self.value_function_worker]
                    else:
                        self.policy_results = [
                            self.sampler, self.curr_logits, self.value_function_manager, self.value_function_worker]
            else:
                self.policy_results = [
                    self.sampler, self.curr_logits, tf.constant("NA")]


    def compute_manager(self, observation):
        s, g  = self.sess.run(
            self.policy_warmup,
            feed_dict={self.observations: [observation]})
        return s, g


    def compute_Q_fuctions(self, observations, actions):
        Q_function_worker = []
        import numpy as np
        actions = np.array(actions)
        means = np.mean(actions, axis=0)
        for j in range(actions.shape[1]):
            actions_j = np.copy(actions)
            actions_j[:, j] = means[j]
            Q_function_worker.append(self.sess.run(
                self.Q_function_worker,
                feed_dict={self.observations: observations, self.actions: actions_j}))
        return Q_function_worker

    def compute_worker(self, gsum, observation):
        if self.ES:
            if self.ADB:

                action, logprobs = self.sess.run(
                    self.policy_results,
                    feed_dict={self.observations: [observation], self.g_sum: gsum})

                vfw = self.sess.run(
                    self.Q_function_worker,
                    feed_dict={self.observations: [observation], self.actions: action})
            else:
                action, logprobs, vfw = self.sess.run(
                    self.policy_results,
                    feed_dict={self.observations: [observation], self.g_sum: gsum})

            return action[0], {"vf_preds_worker": vfw[0], "logprobs": logprobs[0]}

        else:
            if self.ADB:
                action, logprobs, vfm = self.sess.run(
                    self.policy_results,
                    feed_dict={self.observations: [observation], self.g_sum: gsum})

                vfw = self.sess.run(
                    self.Q_function_worker,
                    feed_dict={self.observations: [observation], self.actions: action})
            else:
                action, logprobs, vfm, vfw = self.sess.run(
                    self.policy_results,
                    feed_dict={self.observations: [observation], self.g_sum: gsum})

            return action[0], {"vf_preds_manager": vfm[0], "vf_preds_worker": vfw[0],"logprobs": logprobs[0]}


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





