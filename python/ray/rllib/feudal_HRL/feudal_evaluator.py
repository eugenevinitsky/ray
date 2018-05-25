from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import tensorflow as tf
import os

from tensorflow.python import debug as tf_debug

import numpy as np

import ray
from ray.rllib.optimizers import PolicyEvaluator, SampleBatch
from ray.rllib.optimizers.multi_gpu_impl import LocalSyncParallelOptimizer_Feudal
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.sampler import SyncSampler_Feudal
from ray.rllib.utils.filter import get_filter, MeanStdFilter
from ray.rllib.utils.process_rollout import process_rollout_Feudal, process_rollout_Feudal_AD



from ray.rllib.feudal_HRL.loss import FeudalLoss

# TODO(rliaw): Move this onto LocalMultiGPUOptimizer
class FeudalEvaluator(PolicyEvaluator):
    """
    Runner class that holds the simulator environment and the policy.

    Initializes the tensorflow graphs for both training and evaluation.
    One common policy graph is initialized on '/cpu:0' and holds all the shared
    network weights. When run as a remote agent, only this graph is used.
    """

    def __init__(self, registry, env_creator, config, logdir, is_remote, ADB):

        self.ADB = ADB
        self.registry = registry
        self.is_remote = is_remote
        if is_remote:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            devices = ["/cpu:0"]
        else:
            devices = config["devices"]
        self.devices = devices
        self.config = config
        self.logdir = logdir
        self.env = ModelCatalog.get_preprocessor_as_wrapper(
            registry, env_creator(config["env_config"]), config["model"])
        if is_remote:
            config_proto = tf.ConfigProto()
        else:
            config_proto = tf.ConfigProto(**config["tf_session_args"])
        self.sess = tf.Session(config=config_proto)
        if config["tf_debug_inf_or_nan"] and not is_remote:
            self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
            self.sess.add_tensor_filter(
                "has_inf_or_nan", tf_debug.has_inf_or_nan)

        # Defines the training inputs:
        # The coefficient of the KL penalty.
        shared_model = (self.config["model"].get("custom_options", {}).
                        get("multiagent_shared_model", False))
        if shared_model:
            num_kl_terms = 1
        else:
            num_kl_terms = len(self.config["model"].get("custom_options", {}).
                               get("multiagent_obs_shapes", [1]))
        self.kl_coeff = tf.placeholder(
            name="newkl", shape=(num_kl_terms,), dtype=tf.float32)

        obs_space = self.env.observation_space
        action_space = self.env.action_space
        action_dim = action_space.shape[0]
        # The input observations.

        self.diff = tf.placeholder(
            tf.float32, shape=(None, self.config["g_dim"]))
        self.gsum = tf.placeholder(
            tf.float32, shape=(None, self.config["g_dim"]))
        self.observations = tf.placeholder(
            tf.float32, shape=(None, ) + obs_space.shape)

        # Targets of the value functions.
        self.value_targets_manager = tf.placeholder(tf.float32, shape=(None,))
        self.value_targets_worker = tf.placeholder(tf.float32, shape=(None,))
        # Advantage values in the policy gradient estimator.

        self.advantages_manager = tf.placeholder(tf.float32, shape=(None,))

        if self.config["ADB"]:
            self.advantages_worker = tf.placeholder(tf.float32, shape=(None, action_dim))
        else:
            self.advantages_worker = tf.placeholder(tf.float32, shape=(None,))

        self.actions = ModelCatalog.get_action_placeholder(action_space)
        self.distribution_class_obs, self.obs_dim = ModelCatalog.get_obs_dist(obs_space)
        self.distribution_class, self.logit_dim = ModelCatalog.get_action_dist(
            action_space)
        # Log probabilities from the policy before the policy update.
        self.prev_logits = tf.placeholder(
            tf.float32, shape=(None, self.logit_dim))
        # Value functions predictions before the policy update.
        self.prev_vf_preds_manager = tf.placeholder(tf.float32, shape=(None,))
        self.prev_vf_preds_worker = tf.placeholder(tf.float32, shape=(None,))

        if is_remote:
            self.batch_size = config["rollout_batchsize"]
            self.per_device_batch_size = config["rollout_batchsize"]
        else:
            self.batch_size = int(
                config["sgd_batchsize"] / len(devices)) * len(devices)
            assert self.batch_size % len(devices) == 0
            self.per_device_batch_size = int(self.batch_size / len(devices))

        def build_loss(diff, gsum, obs, value_targets_manager, value_targets_worker, advantages_manager, advantages_worker,
                           acts,
                           plog, prev_vf_preds_manager, prev_vf_preds_worker):
                return FeudalLoss(diff, gsum,
                                  self.env.observation_space, self.env.action_space,
                                  obs, value_targets_manager, value_targets_worker, advantages_manager, advantages_worker, acts,
                                  plog, prev_vf_preds_manager, prev_vf_preds_worker, self.logit_dim,
                                  self.kl_coeff, self.distribution_class, self.distribution_class_obs, self.config,
                                  self.sess, self.registry, self.ADB)



        self.par_opt = LocalSyncParallelOptimizer_Feudal(
            tf.train.AdamOptimizer(self.config["sgd_stepsize"]),
            self.config["num_sgd_iter_baseline_manager"], self.config["num_sgd_iter_baseline_worker"],
            self.devices,
            [self.diff, self.gsum, self.observations, self.value_targets_manager, self.value_targets_worker,
             self.advantages_manager, self.advantages_worker,
             self.actions, self.prev_logits, self.prev_vf_preds_manager, self.prev_vf_preds_worker],
            self.per_device_batch_size,
            build_loss,
            self.logdir)

        # Metric ops
        if self.config["num_sgd_iter_baseline_manager"] == 0 and self.config["num_sgd_iter_baseline_worker"] == 0:
            with tf.name_scope("test_outputs"):
                policies_manager, policies_worker = self.par_opt.get_device_losses()
                self.loss_manager = tf.reduce_mean(
                    tf.stack(values=[
                        policy.loss_manager for policy in policies_manager]), 0)
                self.mean_vf_loss_manager = tf.reduce_mean(
                    tf.stack(values=[
                        policy.mean_vf_loss_manager for policy in policies_manager]), 0)
                self.manager_policy_loss = tf.reduce_mean(
                    tf.stack(values=[
                        policy.mean_surr_manager for policy in policies_manager]), 0)
                self.loss_worker = tf.reduce_mean(
                    tf.stack(values=[
                        policy.loss_worker for policy in policies_worker]), 0)
                self.mean_policy_loss_worker = tf.reduce_mean(
                    tf.stack(values=[
                        policy.mean_policy_loss_worker for policy in policies_worker]), 0)
                self.mean_vf_loss_worker = tf.reduce_mean(
                    tf.stack(values=[
                        policy.mean_vf_loss_worker for policy in policies_worker]), 0)
                self.mean_kl = tf.reduce_mean(
                    tf.stack(values=[
                        policy.mean_kl for policy in policies_worker]), 0)
                self.mean_entropy_worker = tf.reduce_mean(
                    tf.stack(values=[
                        policy.mean_entropy_worker for policy in policies_worker]), 0)

        elif self.config["num_sgd_iter_baseline_manager"] > 0 and self.config["num_sgd_iter_baseline_worker"] == 0:

            policies_manager_loss, policies_manager_baseline, policies_worker = self.par_opt.get_device_losses()
            self.loss_manager = tf.reduce_mean(
                tf.stack(values=[
                    policy.loss_manager for policy in policies_manager_loss]), 0)
            self.mean_vf_loss_manager = tf.reduce_mean(
                tf.stack(values=[
                    policy.mean_vf_loss_manager for policy in policies_manager_baseline]), 0)
            self.manager_policy_loss = tf.reduce_mean(
                tf.stack(values=[
                    policy.mean_surr_manager for policy in policies_manager_loss]), 0)
            self.loss_worker = tf.reduce_mean(
                tf.stack(values=[
                    policy.loss_worker for policy in policies_worker]), 0)
            self.mean_policy_loss_worker = tf.reduce_mean(
                tf.stack(values=[
                    policy.mean_policy_loss_worker for policy in policies_worker]), 0)
            self.mean_vf_loss_worker = tf.reduce_mean(
                tf.stack(values=[
                    policy.mean_vf_loss_worker for policy in policies_worker]), 0)
            self.mean_kl = tf.reduce_mean(
                tf.stack(values=[
                    policy.mean_kl for policy in policies_worker]), 0)
            self.mean_entropy_worker = tf.reduce_mean(
                tf.stack(values=[
                    policy.mean_entropy_worker for policy in policies_worker]), 0)
        elif self.config["num_sgd_iter_baseline_manager"] == 0 and self.config["num_sgd_iter_baseline_worker"] > 0:
            policies_manager, policies_worker, policies_worker_baseline = self.par_opt.get_device_losses()
            self.loss_manager = tf.reduce_mean(
                tf.stack(values=[
                    policy.loss_manager for policy in policies_manager]), 0)
            self.mean_vf_loss_manager = tf.reduce_mean(
                tf.stack(values=[
                    policy.mean_vf_loss_manager for policy in policies_manager]), 0)
            self.manager_policy_loss = tf.reduce_mean(
                tf.stack(values=[
                    policy.mean_surr_manager for policy in policies_manager]), 0)
            self.loss_worker = tf.reduce_mean(
                tf.stack(values=[
                    policy.loss_worker for policy in policies_worker]), 0)
            self.mean_policy_loss_worker = tf.reduce_mean(
                tf.stack(values=[
                    policy.mean_policy_loss_worker for policy in policies_worker]), 0)
            self.mean_vf_loss_worker = tf.reduce_mean(
                tf.stack(values=[
                    policy.mean_vf_loss_worker for policy in policies_worker_baseline]), 0)
            self.mean_kl = tf.reduce_mean(
                tf.stack(values=[
                    policy.mean_kl for policy in policies_worker]), 0)
            self.mean_entropy_worker = tf.reduce_mean(
                tf.stack(values=[
                    policy.mean_entropy_worker for policy in policies_worker]), 0)
        else:
            policies_manager_loss, policies_manager_baseline, policies_worker, policies_worker_baseline = self.par_opt.get_device_losses()
            self.loss_manager = tf.reduce_mean(
                tf.stack(values=[
                    policy.loss_manager for policy in policies_manager_loss]), 0)
            self.mean_vf_loss_manager = tf.reduce_mean(
                tf.stack(values=[
                    policy.mean_vf_loss_manager for policy in policies_manager_baseline]), 0)
            self.manager_policy_loss = tf.reduce_mean(
                tf.stack(values=[
                    policy.mean_surr_manager for policy in policies_manager_loss]), 0)
            self.loss_worker = tf.reduce_mean(
                tf.stack(values=[
                    policy.loss_worker for policy in policies_worker]), 0)
            self.mean_policy_loss_worker = tf.reduce_mean(
                tf.stack(values=[
                    policy.mean_policy_loss_worker for policy in policies_worker]), 0)
            self.mean_vf_loss_worker = tf.reduce_mean(
                tf.stack(values=[
                    policy.mean_vf_loss_worker for policy in policies_worker_baseline]), 0)
            self.mean_kl = tf.reduce_mean(
                tf.stack(values=[
                    policy.mean_kl for policy in policies_worker]), 0)
            self.mean_entropy_worker = tf.reduce_mean(
                tf.stack(values=[
                    policy.mean_entropy_worker for policy in policies_worker]), 0)


        # References to the model weights
        self.common_policy = self.par_opt.get_common_loss()

        if self.config["num_sgd_iter_baseline_manager"] == 0 and self.config["num_sgd_iter_baseline_worker"] == 0:
            self.variables_manager_loss = ray.experimental.TensorFlowVariables(
                self.common_policy.loss_manager, self.sess)
            self.variables_worker_loss = ray.experimental.TensorFlowVariables(
                self.common_policy.loss_worker, self.sess)

        elif self.config["num_sgd_iter_baseline_manager"] > 0 and self.config["num_sgd_iter_baseline_worker"] == 0:
            self.variables_manager_loss = ray.experimental.TensorFlowVariables(
                self.common_policy.loss_manager, self.sess)
            self.variables_manager_baseline = ray.experimental.TensorFlowVariables(
                self.common_policy.mean_vf_loss_manager, self.sess)
            self.variables_worker_loss = ray.experimental.TensorFlowVariables(
                self.common_policy.loss_worker, self.sess)

        elif self.config["num_sgd_iter_baseline_manager"] == 0 and self.config["num_sgd_iter_baseline_worker"] > 0:
            self.variables_manager_loss = ray.experimental.TensorFlowVariables(
                self.common_policy.loss_manager, self.sess)
            self.variables_worker_loss = ray.experimental.TensorFlowVariables(
                self.common_policy.loss_worker, self.sess)
            self.variables_worker_baseline = ray.experimental.TensorFlowVariables(
                self.common_policy.mean_vf_loss_worker, self.sess)

        else:
            self.variables_manager_loss = ray.experimental.TensorFlowVariables(
                self.common_policy.loss_manager, self.sess)
            self.variables_manager_baseline = ray.experimental.TensorFlowVariables(
                self.common_policy.mean_vf_loss_manager, self.sess)
            self.variables_worker_loss = ray.experimental.TensorFlowVariables(
                self.common_policy.loss_worker, self.sess)
            self.variables_worker_baseline = ray.experimental.TensorFlowVariables(
                self.common_policy.mean_vf_loss_worker, self.sess)

        self.obs_filter = get_filter(
            config["observation_filter"], self.env.observation_space.shape)
        self.rew_filter = MeanStdFilter((), clip=5.0)
        self.filters = {"obs_filter": self.obs_filter,
                        "rew_filter": self.rew_filter}
        self.sampler = SyncSampler_Feudal(
            self.env, self.common_policy, self.obs_filter,
            self.config["horizon"], self.ADB, self.config["c"], self.config["dilatation_rate"], self.config["horizon"])
        self.sess.run(tf.global_variables_initializer())

    def load_data(self, trajectories, full_trace):
        return self.par_opt.load_data(
            self.sess,
            [trajectories["diff"],
             trajectories["gsum"],
             trajectories["obs"],
             trajectories["value_targets_manager"],
             trajectories["value_targets_worker"],
             trajectories["advantages_manager"],
             trajectories["advantages_worker"],
             trajectories["actions"],
             trajectories["logprobs"],
             trajectories["vf_preds_manager"],
             trajectories["vf_preds_worker"]],
            full_trace=full_trace)

    def run_sgd_minibatch_manager(
            self, batch_index, full_trace, file_writer):
        return self.par_opt.optimize(
            self.sess,
            batch_index,
            manager=True,
            worker=False,
            baseline_manager=False,
            baseline_worker=False,
            extra_ops=[self.loss_manager,  self.mean_vf_loss_manager, self.manager_policy_loss ],
            file_writer=file_writer if full_trace else None)

    def run_sgd_minibatch_baseline_manager(
            self, batch_index, full_trace, file_writer):
        return self.par_opt.optimize(
            self.sess,
            batch_index,
            manager=False,
            worker=False,
            baseline_manager=True,
            baseline_worker=False,
            extra_ops=[self.mean_vf_loss_manager],
            file_writer=file_writer if full_trace else None)

    def run_sgd_minibatch_worker(self, batch_index, kl_coeff, full_trace, file_writer):
        return self.par_opt.optimize(
            self.sess,
            batch_index,
            manager=False,
            worker=True,
            baseline_manager=False,
            baseline_worker=False,
            extra_ops=[
                self.loss_worker, self.mean_policy_loss_worker, self.mean_vf_loss_worker,
                self.mean_kl, self.mean_entropy_worker],
            extra_feed_dict={self.kl_coeff: kl_coeff},
            file_writer=file_writer if full_trace else None)

    def run_sgd_minibatch_baseline_worker(
            self, batch_index, full_trace, file_writer):
        return self.par_opt.optimize(
            self.sess,
            batch_index,
            manager=False,
            worker=False,
            baseline_manager=False,
            baseline_worker=True,
            extra_ops=[self.mean_vf_loss_worker],
            file_writer=file_writer if full_trace else None)

    def compute_gradients(self, samples):
        raise NotImplementedError

    def apply_gradients(self, grads):
        raise NotImplementedError

    def save(self):
        filters = self.get_filters(flush_after=True)
        return pickle.dumps({"filters": filters})

    def restore(self, objs):
        objs = pickle.loads(objs)
        self.sync_filters(objs["filters"])


    def get_weights_manager_loss(self):
        return self.variables_manager_loss.get_weights()

    def set_weights_manager_loss(self, weights):
        self.variables_manager_loss.set_weights(weights)


    def get_weights_manager_baseline(self):
        return self.variables_manager_baseline.get_weights()

    def set_weights_manager_baseline(self, weights):
        self.variables_manager_baseline.set_weights(weights)


    def get_weights_worker_loss(self):
        return self.variables_worker_loss.get_weights()

    def set_weights_worker_loss(self, weights):
        self.variables_worker_loss.set_weights(weights)


    def get_weights_worker_baseline(self):
        return self.variables_worker_baseline.get_weights()

    def set_weights_worker_baseline(self, weights):
        self.variables_worker_baseline.set_weights(weights)


    def sample(self):
        """Returns experience samples from this Evaluator. Observation
        filter and reward filters are flushed here.

        Returns:
            SampleBatch: A columnar batch of experiences.
        """
        num_steps_so_far = 0
        all_samples = []

        while num_steps_so_far < self.config["min_steps_per_task"]:
            rollout = self.sampler.get_data()
            if self.ADB:
                samples = process_rollout_Feudal_AD(self.config["c"], self.config["tradeoff_rewards"],
                                                 rollout, self.rew_filter, self.config["gamma"],
                                                 self.config["lambda"], use_gae=self.config["use_gae"])
            else:
                samples = process_rollout_Feudal(self.config["c"], self.config["tradeoff_rewards"],
                                                 rollout, self.rew_filter, self.config["gamma"],
                                                 self.config["lambda"], use_gae=self.config["use_gae"])

            num_steps_so_far += samples.count
            all_samples.append(samples)
        return SampleBatch.concat_samples(all_samples)

    def get_completed_rollout_metrics(self):
        """Returns metrics on previously completed rollouts.

        Calling this clears the queue of completed rollout metrics.
        """
        return self.sampler.get_metrics()

    def sync_filters(self, new_filters):
        """Changes self's filter to given and rebases any accumulated delta.

        Args:
            new_filters (dict): Filters with new state to update local copy.
        """
        assert all(k in new_filters for k in self.filters)
        for k in self.filters:
            self.filters[k].sync(new_filters[k])

    def get_filters(self, flush_after=False):
        """Returns a snapshot of filters.

        Args:
            flush_after (bool): Clears the filter buffer state.

        Returns:
            return_filters (dict): Dict for serializable filters
        """
        return_filters = {}
        for k, f in self.filters.items():
            return_filters[k] = f.as_serializable()
            if flush_after:
                f.clear_buffer()
        return return_filters

