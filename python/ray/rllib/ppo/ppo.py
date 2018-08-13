from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import pickle
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from ray.rllib.utils import log_histogram


import ray
from ray.tune.result import TrainingResult
from ray.tune.trial import Resources
from ray.rllib.agent import Agent
from ray.rllib.utils import FilterManager
from ray.rllib.ppo.ppo_evaluator import PPOEvaluator
from ray.rllib.ppo.rollout import collect_samples


DEFAULT_CONFIG = {
    # Discount factor of the MDP
    "gamma": 0.995,
    # Number of steps after which the rollout gets cut
    "horizon": 2000,
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    "use_gae": True,
    # GAE(lambda) parameter
    "lambda": 1.0,
    # Initial coefficient for KL divergence
    "kl_coeff": 0.2,
    # Number of SGD iterations in each outer loop
    "num_sgd_iter_policy": 30,
    "num_sgd_iter_vf": 30,
    # Stepsize of SGD
    "sgd_stepsize_policy": 5e-5,
    "sgd_stepsize_vf": 5e-5,
    # TODO(pcm): Expose the choice between gpus and cpus
    # as a command line argument.
    "devices": ["/cpu:%d" % i for i in range(4)],
    "tf_session_args": {
        "device_count": {"CPU": 4},
        "log_device_placement": False,
        "allow_soft_placement": True,
        "intra_op_parallelism_threads": 1,
        "inter_op_parallelism_threads": 2,
    },
    # Batch size for policy evaluations for rollouts
    "rollout_batchsize": 1,
    # Total SGD batch size across all devices for SGD
    "sgd_batchsize": 128,
    # Coefficient of the value function loss
    "vf_loss_coeff": 1.0,
    # Coefficient of the entropy regularizer
    "entropy_coeff": 0.0,
    # PPO clip parameter
    "clip_param": 0.3,
    # Target value for KL divergence
    "kl_target": 0.01,
    # Config params to pass to the model
    "model": {"free_log_std": False},
    # Which observation filter to apply to the observation
    "observation_filter": "MeanStdFilter",
    # If >1, adds frameskip
    "extra_frameskip": 1,
    # Number of timesteps collected in each outer loop
    "timesteps_per_batch_policy": 4000,
    "timesteps_per_batch_vf": 4000,
    # Each tasks performs rollouts until at least this
    # number of steps is obtained
    "min_steps_per_task": 200,
    # Number of actors used to collect the rollouts
    "num_workers": 5,
    # Whether to allocate GPUs for workers (if > 0).
    "num_gpus_per_worker": 0,
    # Whether to allocate CPUs for workers (if > 0).
    "num_cpus_per_worker": 1,
    # Dump TensorFlow timeline after this many SGD minibatches
    "full_trace_nth_sgd_batch": -1,
    # Whether to profile data loading
    "full_trace_data_load": False,
    # Outer loop iteration index when we drop into the TensorFlow debugger
    "tf_debug_iteration": -1,
    # If this is True, the TensorFlow debugger is invoked if an Inf or NaN
    # is detected
    "tf_debug_inf_or_nan": False,
    # If True, we write tensorflow logs and checkpoints
    "write_logs": True,
    # Arguments to pass to the env creator
    "env_config": {},
    "regularization_factor": 0,
    "ADB": False,
    "dist_type": None
}


class PPOAgent(Agent):
    _agent_name = "PPO"
    _allow_unknown_subkeys = ["model", "tf_session_args", "env_config"]
    _default_config = DEFAULT_CONFIG

    @classmethod
    def default_resource_request(cls, config):
        cf = dict(cls._default_config, **config)
        return Resources(
            cpu=1,
            gpu=len([d for d in cf["devices"] if "gpu" in d.lower()]),
            extra_cpu=cf["num_cpus_per_worker"] * cf["num_workers"],
            extra_gpu=cf["num_gpus_per_worker"] * cf["num_workers"])

    def _init(self):
        self.ADB = self.config["ADB"]
        self.global_step = 0
        self.global_step_policy = 0
        self.global_step_vf = 0
        self.kl_coeff = self.config["kl_coeff"]
        self.local_evaluator = PPOEvaluator(
            self.registry, self.env_creator, self.config, self.logdir, False, self.ADB)
        RemotePPOEvaluator = ray.remote(
            num_cpus=self.config["num_cpus_per_worker"],
            num_gpus=self.config["num_gpus_per_worker"])(PPOEvaluator)
        self.remote_evaluators = [
            RemotePPOEvaluator.remote(
                self.registry, self.env_creator, self.config, self.logdir,
                True, self.ADB)
            for _ in range(self.config["num_workers"])]
        self.start_time = time.time()
        if self.config["write_logs"]:
            self.file_writer = tf.summary.FileWriter(
                self.logdir, self.local_evaluator.sess.graph)
        else:
            self.file_writer = None
        self.saver = tf.train.Saver(max_to_keep=None)

    def _train(self):
        agents = self.remote_evaluators
        config = self.config
        model = self.local_evaluator

        print("===> iteration", self.iteration)

        iter_start = time.time()
        weights_policy = ray.put(model.get_weights_policy())
        [a.set_weights_policy.remote(weights_policy) for a in agents]
        weights_vf = ray.put(model.get_weights_vf())
        [a.set_weights_vf.remote(weights_vf) for a in agents]


        samples_policy = collect_samples(agents, config, self.local_evaluator, extra_samples=False)

        def standardized(value):
            # Divide by the maximum of value.std() and 1e-4
            # to guard against the case where all values are equal
            return (value - value.mean()) / max(1e-4, value.std())

        samples_policy.data["advantages"] = standardized(samples_policy["advantages"])

        if config["timesteps_per_batch_vf"] > 0:
            print("Taking extra samples")
            extra_samples = collect_samples(agents, config, self.local_evaluator, extra_samples=True)
            extra_samples.data["advantages"] = standardized(extra_samples["advantages"])
            samples_vf = samples_policy.concat(extra_samples)
        else:
            samples_vf = samples_policy.copy()

        rollouts_end = time.time()
        print("Computing policy (iterations=" + str(config["num_sgd_iter_policy"]) +
              ", stepsize=" + str(config["sgd_stepsize_policy"]) + "):")
        names_policy = [
            "iter", "policy loss", "kl", "entropy", "saturation_rate_up", "saturation_rate_down"]
        print(("{:>15}" * len(names_policy)).format(*names_policy))
        samples_policy.shuffle()
        shuffle_end = time.time()
        tuples_per_device = model.load_data(
            samples_policy, self.iteration == 0 and config["full_trace_data_load"])
        load_end = time.time()
        rollouts_time = rollouts_end - iter_start
        shuffle_time = shuffle_end - rollouts_end
        load_time = load_end - shuffle_end
        sgd_time = 0
        for i in range(config["num_sgd_iter_policy"]):
            sgd_start = time.time()
            batch_index = 0
            num_batches = (
                int(tuples_per_device) // int(model.per_device_batch_size))
            policy_graph, kl, entropy, saturation_rate_up, saturation_rate_down  = [], [], [], [], []
            permutation = np.random.permutation(num_batches)
            # Prepare to drop into the debugger
            if self.iteration == config["tf_debug_iteration"]:
                model.sess = tf_debug.LocalCLIDebugWrapperSession(model.sess)
            while batch_index < num_batches:
                full_trace = (
                    i == 0 and self.iteration == 0 and
                    batch_index == config["full_trace_nth_sgd_batch"])
                batch_loss, batch_kl, batch_entropy, batch_saturation_rate_up, batch_saturation_rate_down, clipping = model.run_sgd_minibatch(
                        permutation[batch_index] * model.per_device_batch_size,
                        self.kl_coeff, full_trace,
                        self.file_writer)
                policy_graph.append(batch_loss)
                kl.append(batch_kl)
                entropy.append(batch_entropy)
                saturation_rate_down.append(batch_saturation_rate_down)
                saturation_rate_up.append(batch_saturation_rate_up)
                batch_index += 1
            policy_graph = np.mean(policy_graph)
            kl = np.mean(kl)
            entropy = np.mean(entropy)
            saturation_rate_down = np.mean(saturation_rate_down)
            saturation_rate_up = np.mean(saturation_rate_up)
            sgd_end = time.time()
            print(
                "{:>15}{:15.5e}{:15.5e}{:15.5e}{:15.5e}{:15.5e}".format(
                    i, policy_graph, kl, entropy, saturation_rate_up, saturation_rate_down))
            values_policy = []
            if i == config["num_sgd_iter_policy"] - 1:
                metric_prefix = "ppo/sgd/final_iter/"
                values_policy.append(tf.Summary.Value(
                    tag=metric_prefix + "kl_coeff",
                    simple_value=self.kl_coeff))
                values_policy.extend([
                    tf.Summary.Value(
                        tag=metric_prefix + "mean_entropy",
                        simple_value=entropy),
                    tf.Summary.Value(
                        tag=metric_prefix + "saturation_rate_up",
                        simple_value=saturation_rate_up),
                    tf.Summary.Value(
                        tag=metric_prefix + "saturation_rate_down",
                        simple_value=saturation_rate_down),
                    tf.Summary.Value(
                        tag=metric_prefix + "mean_loss",
                        simple_value=policy_graph),
                    tf.Summary.Value(
                        tag=metric_prefix + "mean_kl",
                        simple_value=kl)])

                if self.file_writer:
                    sgd_stats = tf.Summary(value=values_policy)
                    self.file_writer.add_summary(sgd_stats, self.global_step_policy)
                    weights_policy = self.local_evaluator.get_weights_policy()
                    for key, variable in weights_policy.items():
                        log_histogram.log_histogram(self.file_writer, key, variable, self.global_step_policy)

            self.global_step_policy += 1
            sgd_time += sgd_end - sgd_start

            print("clipping")
            print(clipping)

        print("Fitting the baseline")

        print("Computing policy (iterations=" + str(config["num_sgd_iter_vf"]) +
              ", stepsize=" + str(config["sgd_stepsize_vf"]) + "):")
        names_vf = [
            "iter", "vf_loss"]
        print(("{:>15}" * len(names_vf)).format(*names_vf))
        tuples_per_device_vf = model.load_data(
            samples_vf, self.iteration == 0 and config["full_trace_data_load"])
        for i in range(config["num_sgd_iter_vf"]):
            batch_index = 0
            num_batches = (
                    int(tuples_per_device_vf) // int(model.per_device_batch_size))
            vf_loss = []
            permutation = np.random.permutation(num_batches)
            # Prepare to drop into the debugger
            if self.iteration == config["tf_debug_iteration"]:
                model.sess = tf_debug.LocalCLIDebugWrapperSession(model.sess)
            while batch_index < num_batches:
                full_trace = (
                        i == 0 and self.iteration == 0 and
                        batch_index == config["full_trace_nth_sgd_batch"])
                batch_vf_loss = model.run_sgd_minibatch_baseline(
                    permutation[batch_index] * model.per_device_batch_size,
                    full_trace,
                    self.file_writer)
                vf_loss.append(batch_vf_loss)
                batch_index += 1
            vf_loss = np.mean(vf_loss)
            print(
                "{:>15}{:15.5e}".format(
                    i, vf_loss))
            values_vf = []
            if i == config["num_sgd_iter_vf"] - 1:
                metric_prefix = "ppo/sgd/final_iter/"
                values_vf.append(tf.Summary.Value(
                    tag=metric_prefix + "vf_loss",
                    simple_value=vf_loss))

                if self.file_writer:
                    sgd_stats = tf.Summary(value=values_vf)
                    self.file_writer.add_summary(sgd_stats, self.global_step_vf)
                    weights_vf = self.local_evaluator.get_weights_vf()
                    for key, variable in weights_vf.items():
                        log_histogram.log_histogram(self.file_writer, key, variable, self.global_step_vf)
            self.global_step_vf+=1

        if kl > 2.0 * config["kl_target"]:
            self.kl_coeff *= 1.5
        elif kl < 0.5 * config["kl_target"]:
            self.kl_coeff *= 0.5

        info = {
            "kl_divergence": kl,
            "kl_coefficient": self.kl_coeff,
            "rollouts_time": rollouts_time,
            "shuffle_time": shuffle_time,
            "load_time": load_time,
            "sgd_time": sgd_time,
            "sample_throughput": len(samples_policy["obs"]) / sgd_time
        }
        self.global_step+=1

        FilterManager.synchronize(
            self.local_evaluator.filters, self.remote_evaluators)
        res = self._fetch_metrics_from_remote_evaluators()
        res = res._replace(info=info)
        return res

    def _fetch_metrics_from_remote_evaluators(self):
        episode_rewards = []
        episode_lengths = []
        metric_lists = [a.get_completed_rollout_metrics.remote()
                        for a in self.remote_evaluators]
        for metrics in metric_lists:
            for episode in ray.get(metrics):
                episode_lengths.append(episode.episode_length)
                episode_rewards.append(episode.episode_reward)
        avg_reward = (
            np.mean(episode_rewards) if episode_rewards else float('nan'))
        avg_length = (
            np.mean(episode_lengths) if episode_lengths else float('nan'))
        timesteps = np.sum(episode_lengths) if episode_lengths else 0

        result = TrainingResult(
            episode_reward_mean=avg_reward,
            episode_len_mean=avg_length,
            timesteps_this_iter=timesteps)

        return result

    def _stop(self):
        # workaround for https://github.com/ray-project/ray/issues/1516
        for ev in self.remote_evaluators:
            ev.__ray_terminate__.remote(ev._ray_actor_id.id())

    def _save(self, checkpoint_dir):
        checkpoint_path = self.saver.save(
            self.local_evaluator.sess,
            os.path.join(checkpoint_dir, "checkpoint"),
            global_step=self.iteration)
        agent_state = ray.get(
            [a.save.remote() for a in self.remote_evaluators])
        extra_data = [
            self.local_evaluator.save(),
            self.global_step,
            self.kl_coeff,
            agent_state]
        pickle.dump(extra_data, open(checkpoint_path + ".extra_data", "wb"))
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.saver.restore(self.local_evaluator.sess, checkpoint_path)
        extra_data = pickle.load(open(checkpoint_path + ".extra_data", "rb"))
        self.local_evaluator.restore(extra_data[0])
        self.global_step = extra_data[1]
        self.kl_coeff = extra_data[2]
        ray.get([
            a.restore.remote(o)
                for (a, o) in zip(self.remote_evaluators, extra_data[3])])

    def compute_action(self, observation):
        observation = self.local_evaluator.obs_filter(
            observation, update=False)
        return self.local_evaluator.common_policy.compute_single_action(
            observation, [], False)[0]