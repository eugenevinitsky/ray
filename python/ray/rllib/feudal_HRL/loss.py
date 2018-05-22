from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ray.rllib.models import ModelCatalog
from ray.rllib.models.singlestepLSTM import SingleStepLSTM


class FeudalLoss(object):

    other_output = ["vf_preds", "logprobs"]
    is_recurrent = False

    def __init__(
            self, observation_space, action_space,
            observations, value_targets_manager, value_targets_worker, advantages_manager, advantages_worker, actions,
            prev_logits, prev_vf_preds_manager, prev_vf_preds_worker, logit_dim,
            kl_coeff, distribution_class, distribution_class_obs, config, sess, registry):


        self.prev_dist = distribution_class(prev_logits)
        self.shared_model = (config["model"].get("custom_options", {}).
                             get("multiagent_shared_model", False))
        self.num_agents = len(config["model"].get(
            "custom_options", {}).get("multiagent_obs_shapes", [1]))

        # Saved so that we can compute actions given different observations
        self.observations = observations

        with tf.variable_scope("z"):
            self.z = tf.layers.dense(inputs=observations, \
                                     units=256, \
                                     activation=tf.nn.relu)


        with tf.variable_scope("Manager"):
            self.s = tf.layers.dense(inputs=self.z, \
                                 units=config["g_dim"], \
                                 activation=tf.nn.elu)

            self.diff = self.s[config["c"]:] - self.s[:-config["c"]]
            tensor = [tf.reshape(self.s[-1] - self.s[-1 - config["c"] + i], shape=(1, self.s[-1].shape[0])) for i in range(config["c"])]
            tensor = tf.concat(tensor, axis=0)
            self.diff = tf.concat([self.diff, tensor], axis=0)
            x = tf.expand_dims(self.s, [0])
            self.manager_lstm = SingleStepLSTM(x, config["g_dim"], \
                                               step_size=tf.shape(self.observations)[:1])
            g_hat = self.manager_lstm.output
            self.g = tf.nn.l2_normalize(g_hat, dim=1)
            self.manager_logits = distribution_class_obs(self.g, config["kappa"], observation_space.shape[0])



            """BASELINE"""
            vf_config = config["model"].copy()
            # Do not split the last layer of the value function into
            # mean parameters and standard deviation parameters and
            # do not make the standard deviations free variables.
            vf_config["free_log_std"] = False
            with tf.variable_scope("value_function_manager"):
                self.value_function_manager = ModelCatalog.get_model(
                    registry, observations, 1, vf_config).outputs
            self.value_function_manager = tf.reshape(self.value_function_manager, [-1])

            self.logp_manager = self.manager_logits.logp(self.diff)

            if not isinstance(self.logp_manager, list):
                self.logp_manager = [self.logp_manager]
                #self.entropy = [self.entropy]

            self.logp_manager = [logp for logp in zip(self.logp_manager)]
            print(self.logp_manager)
            print("self.logp_manager")
            self.surr_manager = [logp_manager * advantages_manager for logp_manager in self.logp_manager]
            self.surr_manager = tf.add_n(self.surr_manager)
            self.surr_manager = tf.reduce_mean(self.surr_manager)

            self.vf_loss1_manager = tf.square(self.value_function_manager - value_targets_manager)
            vf_clipped_manager = prev_vf_preds_manager + tf.clip_by_value(
                self.value_function_manager - prev_vf_preds_manager,
                -config["clip_param"], config["clip_param"])
            self.vf_loss2_manager = tf.square(vf_clipped_manager - value_targets_manager)
            self.vf_loss_manager = tf.minimum(self.vf_loss1_manager, self.vf_loss2_manager)
            self.mean_vf_loss_manager = tf.reduce_mean(self.vf_loss_manager)

            self.loss_manager = self.loss = tf.reduce_mean(
                        -self.surr_manager +
                        config["vf_loss_coeff_manager"] * self.vf_loss_manager)



        with tf.variable_scope("Worker"):

            vf_config = config["model"].copy()
            # Do not split the last layer of the value function into
            # mean parameters and standard deviation parameters and
            # do not make the standard deviations free variables.
            vf_config["free_log_std"] = False
            with tf.variable_scope("value_function_worker"):
                self.value_function_worker = ModelCatalog.get_model(
                    registry, observations, 1, vf_config).outputs
            self.value_function_worker = tf.reshape(self.value_function_worker, [-1])

            num_acts = action_space.shape[0]

            # Calculate U
            self.worker_lstm = SingleStepLSTM(tf.expand_dims(self.z, [0]), \
                                              size=2 * num_acts * config["k"],
                                              step_size=tf.shape(self.observations)[:1])
            flat_logits = self.worker_lstm.output
            U = tf.reshape(flat_logits, [-1, 2 * num_acts, config["k"]])
            # Calculate w
            cut_g = tf.stop_gradient(self.g)
            gsum = 0
            for i in range(config["c"]):
                print("gsum")
                print(gsum)
                print("i")
                print(i)
                zeros = tf.zeros(shape=(i, config["g_dim"]), dtype=tf.float32)
                tensor = [tf.reshape(cut_g[i], shape=(1,cut_g[i].shape[0])) for _ in range(config["c"] - i)]
                tensor = tf.concat(tensor, axis=0)
                tensor = tf.concat([zeros, tensor], axis=0)
                print("cut_g[i:-i]")
                print(cut_g[i:-i])
                print("tensor")
                print(tensor)
                gsum +=  tf.concat([tensor, cut_g[i:-i]], axis=0)

            phi = tf.get_variable("phi", (config["g_dim"], config['k']))
            w = tf.matmul(gsum, phi)
            w = tf.expand_dims(w, [2])
            # Calculate policy and sample
            self.curr_logits = tf.reshape(tf.matmul(U, w), [-1, 2 * num_acts])
            self.curr_dist = distribution_class(self.curr_logits)
            self.sampler = self.curr_dist.sample()

            curr_logp = self.curr_dist.logp(actions)
            prev_logp = self.prev_dist.logp(actions)
            self.kl = self.prev_dist.kl(self.curr_dist)
            self.entropy_worker = self.curr_dist.entropy()

            # handle everything uniform as if it were the multiagent case
            if not isinstance(curr_logp, list):
                self.kl = [self.kl]
                curr_logp = [curr_logp]
                prev_logp = [prev_logp]
                self.entropy_worker = [self.entropy_worker]

            # Make loss functions.
            self.ratio_worker = [tf.exp(curr - prev)
                          for curr, prev in zip(curr_logp, prev_logp)]
            self.mean_kl = [tf.reduce_mean(kl_i) for kl_i in self.kl]
            self.mean_entropy_worker = tf.reduce_mean(self.entropy_worker)
            self.surr1_worker = [ratio_i * advantages_worker for ratio_i in self.ratio_worker]
            self.surr2_worker = [tf.clip_by_value(ratio_i, 1 - config["clip_param"],
                                           1 + config["clip_param"]) * advantages_worker
                          for ratio_i in self.ratio_worker]
            self.surr_worker = [tf.minimum(surr1_i, surr2_i) for surr1_i, surr2_i in
                         zip(self.surr1_worker, self.surr2_worker)]
            self.surr_worker = tf.add_n(self.surr_worker)
            self.mean_policy_loss_worker = tf.reduce_mean(-self.surr_worker)

            self.entropy_worker = tf.add_n(self.entropy_worker)
            entropy_prod_worker = config["entropy_coeff"]*self.entropy_worker

            # there's only one kl value for a shared model
            if self.shared_model:
                kl_prod = tf.add_n([kl_coeff[0] * kl_i for
                                    i, kl_i in enumerate(self.kl)])
                # all the above values have been rescaled by num_agents
                self.surr_worker /= self.num_agents
                kl_prod /= self.num_agents
                entropy_prod_worker /= self.num_agents
            else:
                kl_prod = tf.add_n([kl_coeff[i] * kl_i for
                                    i, kl_i in enumerate(self.kl)])

            if config["use_gae"]:
                # We use a huber loss here to be more robust against outliers,
                # which seem to occur when the rollouts get longer (the variance
                # scales superlinearly with the length of the rollout)
                self.vf_loss1_worker= tf.square(self.value_function_worker - value_targets_worker)
                vf_clipped_worker = prev_vf_preds_worker + tf.clip_by_value(
                    self.value_function_worker - prev_vf_preds_worker,
                    -config["clip_param"], config["clip_param"])
                self.vf_loss2_worker = tf.square(vf_clipped_worker - value_targets_worker)
                self.vf_loss_worker = tf.minimum(self.vf_loss1_worker, self.vf_loss2_worker)
                self.mean_vf_loss_worker = tf.reduce_mean(self.vf_loss_worker)
                if config["num_sgd_iter_baseline"] == 0:
                    self.loss_worker = tf.reduce_mean(
                        -self.surr_worker + kl_prod +
                        config["vf_loss_coeff_worker"] * self.vf_loss_worker -
                        entropy_prod_worker)
                else:
                    self.loss = tf.reduce_mean(
                        -self.surr_worker + kl_prod -
                        entropy_prod_worker)
            else:
                self.mean_vf_loss_worker = tf.constant(0.0)
                self.loss = tf.reduce_mean(
                    -self.surr +
                    kl_prod - entropy_prod_worker)

            self.sess = sess

            if config["use_gae"]:
                self.policy_results = [
                    self.sampler, self.curr_logits, self.value_function_manager, self.value_function_worker, self.s, self.g]
            else:
                self.policy_results = [
                    self.sampler, self.curr_logits, tf.constant("NA")]

    def compute(self, observation):
            action, logprobs, vfm, vfw, s, g = self.sess.run(
                self.policy_results,
                feed_dict={self.observations: [observation]})
            return action[0], {"vf_preds_manager": vfm[0], "vf_preds_worker": vfw[0],"logprobs": logprobs[0]}, s, g


    def loss_manager(self):
        return self.loss_manager

    def mean_vf_loss_manager(self):
            return self.mean_vf_loss_manager

    def loss_worker(self):
            return self.loss

    def mean_vf_loss_worker(self):
            return self.mean_vf_loss_worker

