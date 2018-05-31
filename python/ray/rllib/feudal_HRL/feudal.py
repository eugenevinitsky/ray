from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import pickle
import tensorflow as tf
from tensorflow.python import debug as tf_debug

import ray
from ray.tune.result import TrainingResult
from ray.rllib.agent import Agent
from ray.rllib.utils import FilterManager
from ray.rllib.feudal_HRL.feudal_evaluator import FeudalEvaluator

from ray.rllib.feudal_HRL.rollout import collect_samples


DEFAULT_CONFIG = {
    # Discount factor of the MDP
    "gamma": 0.995,
    # Discount factor of the worker
    "gamma_internal" : 0.997,
    # Tradeoff
    "tradeoff_rewards" : 0.5,
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
    "num_sgd_iter": 30,
    # Number of SGD iterations in each outer loop FOR FITTING THE BASELINE of the MANAGER. If 0 -> NO fitting of the baseline
    "num_sgd_iter_baseline_manager": 0,
    # Number of SGD iterations in each outer loop FOR FITTING THE BASELINE of the WORKER. If 0 -> NO fitting of the baseline
    "num_sgd_iter_baseline_worker": 0,
    # Stepsize of SGD
    "sgd_stepsize": 5e-5,
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
    "vf_loss_coeff_manager": 1.0,
    # Coefficient of the value function loss
    "vf_loss_coeff_worker": 1.0,
    # Coefficient of the entropy regularizer
    "entropy_coeff": 0.0,
    # PPO clip parameter
    "clip_param": 0.2,
    # Target value for KL divergence
    "kl_target": 0.01,
    # Config params to pass to the model
    "model": {"free_log_std": False},
    # Which observation filter to apply to the observation
    "observation_filter": "MeanStdFilter",
    # If >1, adds frameskip
    "extra_frameskip": 1,
    # Number of timesteps collected in each outer loop
    "timesteps_per_batch": 4000,
    # Each tasks performs rollouts until at least this
    # number of steps is obtained
    "min_steps_per_task": 200,
    # Number of actors used to collect the rollouts
    "num_workers": 5,
    # Resource requirements for remote actors
    "worker_resources": {"num_cpus": None},
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
    # dimension of goals
    "g_dim" : 4,
    # kappa coefficient from the VMF distribution
    "kappa" : 1.0,
    # Dimension of w
    "k" : 4,
    # Vf hidden size
    "vf_hidden_size" : 4,
    # Horizon of the manager
    "c" : 2,
    # Boolean variable if the Worker use ADB
    "ADB" : True,
    # Dilatation rate
    "dilatation_rate" : 5
}


class FeudalAgent(Agent):
    _agent_name = "Feudal"
    _allow_unknown_subkeys = ["model", "tf_session_args", "env_config",
                              "worker_resources"]
    _default_config = DEFAULT_CONFIG

    def _init(self):
        self.shared_model = (self.config["model"].get("custom_options", {}).
                        get("multiagent_shared_model", False))
        if self.shared_model:
            self.num_models = 1
        else:
            self.num_models = len(self.config["model"].get(
                "custom_options", {}).get("multiagent_obs_shapes", [1]))
        self.global_step = 0
        self.kl_coeff = [self.config["kl_coeff"]] * self.num_models

        self.local_evaluator = FeudalEvaluator(
            self.registry, self.env_creator, self.config, self.logdir, False, self.config["ADB"])

        RemoteFeudalEvaluator = ray.remote(
            **self.config["worker_resources"])(FeudalEvaluator)
        self.remote_agents = [
            RemoteFeudalEvaluator.remote(
                self.registry, self.env_creator, self.config, self.logdir,
                True, self.config["ADB"])
            for _ in range(self.config["num_workers"])]
        self.start_time = time.time()
        if self.config["write_logs"]:
            self.file_writer = tf.summary.FileWriter(
                self.logdir, self.local_evaluator.sess.graph)
        else:
            self.file_writer = None
        self.saver = tf.train.Saver(max_to_keep=None)

    def _train(self):
        agents = self.remote_agents
        model = self.local_evaluator
        config = self.config

        if (config["num_workers"] * config["min_steps_per_task"] >
                config["timesteps_per_batch"]):
            print(
                "WARNING: num_workers * min_steps_per_task > "
                "timesteps_per_batch. This means that the output of some "
                "tasks will be wasted. Consider decreasing "
                "min_steps_per_task or increasing timesteps_per_batch.")

        print("===> iteration", self.iteration)

        iter_start = time.time()

        weights_manager_loss = ray.put(model.get_weights_manager_loss())
        [a.set_weights_manager_loss.remote(weights_manager_loss) for a in agents]
        weights_worker_loss = ray.put(model.get_weights_worker_loss())
        [a.set_weights_worker_loss.remote(weights_worker_loss) for a in agents]
        if self.config["num_sgd_iter_baseline_manager"] > 0 :
            weights_manager_baseline = ray.put(model.get_weights_manager_baseline())
            [a.set_weights_manager_baseline.remote(weights_manager_baseline) for a in agents]
        if self.config["num_sgd_iter_baseline_worker"] > 0 :
            weights_worker_baseline = ray.put(model.get_weights_worker_baseline())
            [a.set_weights_worker_baseline.remote(weights_worker_baseline) for a in agents]

        samples = collect_samples(agents, config, self.local_evaluator)

        def standardized(value):
            # Divide by the maximum of value.std() and 1e-4
            # to guard against the case where all values are equal
            return (value - value.mean()) / max(1e-4, value.std())

        samples.data["advantages_manager"] = standardized(samples["advantages_manager"])
        samples.data["advantages_worker"] = standardized(samples["advantages_worker"])

        rollouts_end = time.time()
        print("Computing policy (iterations=" + str(config["num_sgd_iter"]) +
              ", stepsize=" + str(config["sgd_stepsize"]) + "):")
        names = [
            "iter", "loss_manager", "vf_loss_manager", "policy_loss_manager", "loss_worker", "policy_loss_worker", "vf_loss_worker", "kl", "entropy_worker"]
        print(("{:>15}" * len(names)).format(*names))
        samples.shuffle()
        shuffle_end = time.time()
        tuples_per_device = model.load_data(
            samples, self.iteration == 0 and config["full_trace_data_load"])
        load_end = time.time()
        rollouts_time = rollouts_end - iter_start
        shuffle_time = shuffle_end - rollouts_end
        load_time = load_end - shuffle_end
        sgd_time = 0
        for i in range(config["num_sgd_iter"]):
            sgd_start = time.time()
            batch_index = 0
            num_batches = (
                int(tuples_per_device) // int(model.per_device_batch_size))
            loss_manager, vf_loss_manager, policy_loss_manager, loss_worker, policy_loss_worker, vf_loss_worker, kl, entropy_worker = [], [], [], [], [], [], [], []
            permutation = np.random.permutation(num_batches)
            # Prepare to drop into the debugger
            if self.iteration == config["tf_debug_iteration"]:
                model.sess = tf_debug.LocalCLIDebugWrapperSession(model.sess)
            while batch_index < num_batches:
                full_trace = (
                    i == 0 and self.iteration == 0 and
                    batch_index == config["full_trace_nth_sgd_batch"])
                batch_loss_manager, batch_vf_loss_manager,  batch_loss_policy_manager= model.run_sgd_minibatch_manager(
                        permutation[batch_index] * model.per_device_batch_size,
                        full_trace,
                        self.file_writer)
                batch_loss_worker, batch_policy_loss_worker, batch_vf_loss_worker, batch_kl, \
                    batch_entropy_worker = model.run_sgd_minibatch_worker(
                        permutation[batch_index] * model.per_device_batch_size,
                        self.kl_coeff, full_trace,
                        self.file_writer)
                loss_manager.append(batch_loss_manager)
                vf_loss_manager.append(batch_vf_loss_manager)
                policy_loss_manager.append(batch_loss_policy_manager)
                loss_worker.append(batch_loss_worker)
                policy_loss_worker.append(batch_policy_loss_worker)
                vf_loss_worker.append(batch_vf_loss_worker)
                kl.append(batch_kl)
                entropy_worker.append(batch_entropy_worker)
                batch_index += 1
            loss_manager = np.mean(loss_manager)
            vf_loss_manager = np.mean(vf_loss_manager)
            policy_loss_manager = np.mean(policy_loss_manager)
            loss_worker = np.mean(loss_worker)
            policy_loss_worker = np.mean(policy_loss_worker)
            vf_loss_worker= np.mean(vf_loss_worker)
            kl = np.mean(kl)
            entropy_worker = np.mean(entropy_worker)
            sgd_end = time.time()
            print(
                "{:>15}{:15.5e}{:15.5e}{:15.5e}{:15.5e}{:15.5e}{:15.5e}{:15.5e}{:15.5e}".format(
                    i, loss_manager, vf_loss_manager, policy_loss_manager, loss_worker, policy_loss_worker, vf_loss_worker, kl, entropy_worker))

            values = []
            if i == config["num_sgd_iter"] - 1:
                metric_prefix = "HRL/sgd/final_iter/"
                values.append(tf.Summary.Value(
                    tag=metric_prefix + "kl_coeff",
                    simple_value=np.mean(self.kl_coeff)))
                values.extend([
                    tf.Summary.Value(
                        tag=metric_prefix + "mean_entropy",
                        simple_value=entropy_worker),
                    tf.Summary.Value(
                        tag=metric_prefix + "mean_loss_manager",
                        simple_value=loss_manager),
                    tf.Summary.Value(
                        tag=metric_prefix + "mean_loss_worker",
                        simple_value=loss_worker),
                    tf.Summary.Value(
                        tag=metric_prefix + "mean_kl",
                        simple_value=kl)])
                if self.file_writer:
                    sgd_stats = tf.Summary(value=values)
                    self.file_writer.add_summary(sgd_stats, self.global_step)
            self.global_step += 1
            sgd_time += sgd_end - sgd_start

        # treat single-agent as a multi-agent system w/ one agent
        if not isinstance(kl, np.ndarray):
            kl = [kl]

        for i, kl_i in enumerate(kl):
            if kl_i > 2.0 * config["kl_target"]:
                self.kl_coeff[i] *= 1.5
            elif kl_i < 0.5 * config["kl_target"]:
                self.kl_coeff[i] *= 0.5

        info = {
            "kl_divergence": np.mean(kl),
            "kl_coefficient": np.mean(self.kl_coeff),
            "rollouts_time": rollouts_time,
            "shuffle_time": shuffle_time,
            "load_time": load_time,
            "sgd_time": sgd_time,
            "sample_throughput": len(samples["obs"]) / sgd_time
        }

        if self.config["num_sgd_iter_baseline_manager"] > 0:
            print("Fitting the baseline of the Manager")

            for i in range(config["num_sgd_iter_baseline_manager"]):
                batch_index = 0
                num_batches = (
                    int(tuples_per_device) // int(model.per_device_batch_size))
                vf_loss_manager = []
                permutation = np.random.permutation(num_batches)
                # Prepare to drop into the debugger
                if self.iteration == config["tf_debug_iteration"]:
                    model.sess = tf_debug.LocalCLIDebugWrapperSession(model.sess)
                while batch_index < num_batches:
                    full_trace = (
                        i == 0 and self.iteration == 0 and
                        batch_index == config["full_trace_nth_sgd_batch"])
                    batch_vf_loss_manager = model.run_sgd_minibatch_baseline_manager(
                            permutation[batch_index] * model.per_device_batch_size,
                            full_trace,
                            self.file_writer)
                    vf_loss_manager.append(batch_vf_loss_manager)
                    batch_index += 1
                vf_loss_manager = np.mean(vf_loss_manager)
                print(
                    "{:>15}{:15.5e}".format(
                        i,  vf_loss_manager))


        if self.config["num_sgd_iter_baseline_worker"] > 0:
            print("Fitting the baseline of the Worker")

            for i in range(config["num_sgd_iter_baseline_worker"]):
                batch_index = 0
                num_batches = (
                    int(tuples_per_device) // int(model.per_device_batch_size))
                vf_loss_worker = []
                permutation = np.random.permutation(num_batches)
                # Prepare to drop into the debugger
                if self.iteration == config["tf_debug_iteration"]:
                    model.sess = tf_debug.LocalCLIDebugWrapperSession(model.sess)
                while batch_index < num_batches:
                    full_trace = (
                        i == 0 and self.iteration == 0 and
                        batch_index == config["full_trace_nth_sgd_batch"])
                    batch_vf_loss_manager = model.run_sgd_minibatch_baseline_worker(
                            permutation[batch_index] * model.per_device_batch_size,
                            full_trace,
                            self.file_writer)
                    vf_loss_worker.append(batch_vf_loss_manager)
                    batch_index += 1
                vf_loss_worker = np.mean(vf_loss_worker)
                print(
                    "{:>15}{:15.5e}".format(
                        i,  vf_loss_worker))



        FilterManager.synchronize(
            self.local_evaluator.filters, self.remote_agents)
        res = self._fetch_metrics_from_remote_evaluators()
        res = res._replace(info=info)

        return res

    def _fetch_metrics_from_remote_evaluators(self):
        episode_rewards = []
        episode_lengths = []
        metric_lists = [a.get_completed_rollout_metrics.remote()
                        for a in self.remote_agents]
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
        for ev in self.remote_agents:
            ev.__ray_terminate__.remote(ev._ray_actor_id.id())

    def _save(self, checkpoint_dir):
        checkpoint_path = self.saver.save(
            self.local_evaluator.sess,
            os.path.join(checkpoint_dir, "checkpoint"),
            global_step=self.iteration)
        agent_state = ray.get(
            [a.save.remote() for a in self.remote_agents])
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
                for (a, o) in zip(self.remote_agents, extra_data[3])])

    def compute_action(self, observation):
        observation = self.local_evaluator.obs_filter(
            observation, update=False)
        return self.local_evaluator.common_policy.compute(observation)[0]
