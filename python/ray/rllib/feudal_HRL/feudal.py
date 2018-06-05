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
from ray.rllib.feudal_HRL.utils import log_histogram


DEFAULT_CONFIG = {
    # Discount factor of the MDP
    "gamma": 0.995,
    # Discount factor of the worker
    "gamma_internal" : 0.997,
    # Number of steps after which the rollout gets cut
    "horizon": 2000,
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    "use_gae": True,
    # GAE(lambda) parameter
    "lambda": 0.97,
    "model": {"free_log_std": False},
    # Initial coefficient for KL divergence
    "lambda_internal": 0.99,
    # Number of SGD iterations in each outer loop
    "num_sgd_iter": 30,
    # Stepsize of SGD
    "sgd_stepsize": np.exp(np.random.uniform(10**(-4.5), 10**(-3.5))),
    # Entropy coefficient
    "entropy_coeff": np.exp(np.random.uniform(10**(-4), 10**(-3))),
    # Treadeoff rewards
    "tradeoff_rewards": np.random.uniform(0, 1),
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
    # Which observation filter to apply to the observation
    "observation_filter": "NoFilter",
    # If >1, adds frameskip
    "extra_frameskip": 1,
    # Number of timesteps collected in each outer loop
    "timesteps_per_batch": 4000,
    # Each tasks performs rollouts until at least this
    # number of steps is obtained
    "min_steps_per_task": 200,
    # Number of actors used to collect the rollouts
    "num_workers": 6,
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
    "g_dim" : 256,
    # kappa coefficient from the VMF distribution
    "kappa" : 1.0,
    # Dimension of w
    "k" : 16,
    # Vf hidden size
    "vf_hidden_size" : 256,
    # Horizon of the manager
    "c" : 10,
    # Dilatation rate
    "dilatation_rate" : 10,
    # WHETEHR THE MODE MANAGER ES IS ACTIVATED OR NOT
    "ES": False,
    # Standard deviation of the noise for parameters perturbation
    "noise_stdev" : 0.1,
    # Hyperparameter for the manager's weights update rule
    "alpha": 1.0,
    # Size of the perceptrons layer before z
    "units_z": 256
}


class FeudalAgent(Agent):
    _agent_name = "FEUDAL"
    _allow_unknown_subkeys = ["model", "tf_session_args", "env_config",
                              "worker_resources"]
    _default_config = DEFAULT_CONFIG

    def _init(self):
        self.ES = self.config["ES"]
        self.global_step = 0

        self.local_evaluator = FeudalEvaluator(
            self.registry, self.env_creator, self.config, self.logdir, True, self.ES)

        RemoteFeudalEvaluator = ray.remote(
            **self.config["worker_resources"])(FeudalEvaluator)

        """WARNING: 2 times number of workers in the case of ES!!!"""
        if self.config["ES"]:
            self.num_workers = 2 * self.config["num_workers"]
        else:
            self.num_workers = self.config["num_workers"]

        self.remote_agents = [
            RemoteFeudalEvaluator.remote(
                self.registry, self.env_creator, self.config, self.logdir,
                True, self.ES)
            for _ in range(self.num_workers)]
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
        print(config["min_steps_per_task"])
        print(self.num_workers)
        print(config["timesteps_per_batch"])
        if (self.num_workers * config["min_steps_per_task"] >
                config["timesteps_per_batch"]):
            print(
                "WARNING: num_workers * min_steps_per_task > "
                "timesteps_per_batch. This means that the output of some "
                "tasks will be wasted. Consider decreasing "
                "min_steps_per_task or increasing timesteps_per_batch.")

        print("===> iteration", self.iteration)

        iter_start = time.time()
        self.noise_table = [dict() for _ in range(len(agents))]

        if self.ES:
            weights_manager_outputs = model.get_weights_manager_loss()
            print("weights_manager_outputs")
            print(weights_manager_outputs)
            count_1 = 0
            count_2 = 0
            for a in agents:
                index = count_1
                count_1 += 1
                weights_manager_outputs_agent = weights_manager_outputs.copy()
                noise_agent = weights_manager_outputs.copy()
                for key, variable in weights_manager_outputs_agent.items():
                    count_2 += 1
                    seed = count_1 * len(agents) + count_2 * len(key)
                    shape = weights_manager_outputs_agent[key].shape
                    noise_ = np.random.RandomState(seed).normal(loc=0.0, scale=1.0, size=shape).astype(np.float32)

                    """TO REDUCE VARIANCE +epsilon AND -epsilon"""
                    if count_1%2==0:
                        weights_manager_outputs_agent[key] += self.config["noise_stdev"] * noise_
                        noise_agent[key] = noise_
                    else:
                        weights_manager_outputs_agent[key] -= self.config["noise_stdev"] * noise_
                        noise_agent[key] = -noise_

                self.noise_table[index] = noise_agent

                a.set_weights_manager_loss.remote(weights_manager_outputs_agent)


        else:
            weights_manager_loss = ray.put(model.get_weights_manager_loss())
            [a.set_weights_manager_loss.remote(weights_manager_loss) for a in agents]


        weights_worker_loss = ray.put(model.get_weights_worker_loss())
        print("weights_worker_loss")
        print(model.get_weights_worker_loss())
        [a.set_weights_worker_loss.remote(weights_worker_loss) for a in agents]

        samples = collect_samples(agents, config, self.local_evaluator)

        def standardized(value):
            # Divide by the maximum of value.std() and 1e-4
            # to guard against the case where all values are equal
            return (value - value.mean()) / max(1e-4, value.std())

        if self.ES == False:
            samples.data["advantages_manager"] = standardized(samples["advantages_manager"])
        samples.data["advantages_worker"] = standardized(samples["advantages_worker"])

        rollouts_end = time.time()
        print("Computing policy (iterations=" + str(config["num_sgd_iter"]) +
              ", stepsize=" + str(config["sgd_stepsize"]) + "):")
        if self.ES:
            names = [
                "iter", "loss_worker", "policy_loss_worker", "vf_loss_worker", "entropy_worker"]
        else:
            names = [
                "iter", "loss_manager", "vf_loss_manager", "policy_loss_manager", "loss_worker", "policy_loss_worker",
                "vf_loss_worker", "entropy_worker"]

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
            if self.ES:
                loss_worker, policy_loss_worker, vf_loss_worker, entropy_worker = [], [], [], []
            else:
                loss_manager, vf_loss_manager, policy_loss_manager, loss_worker, policy_loss_worker, vf_loss_worker, entropy_worker = [], [], [], [], [], [], []
            permutation = np.random.permutation(num_batches)
            # Prepare to drop into the debugger
            if self.iteration == config["tf_debug_iteration"]:
                model.sess = tf_debug.LocalCLIDebugWrapperSession(model.sess)
            while batch_index < num_batches:
                full_trace = (
                    i == 0 and self.iteration == 0 and
                    batch_index == config["full_trace_nth_sgd_batch"])
                if self.ES == False:
                    batch_loss_manager, batch_vf_loss_manager, batch_loss_policy_manager, batch_loss_worker, batch_policy_loss_worker, batch_vf_loss_worker, \
                    batch_entropy_worker = model.run_sgd_minibatch(
                            permutation[batch_index] * model.per_device_batch_size,
                            full_trace,
                            self.file_writer)
                    loss_manager.append(batch_loss_manager)
                    vf_loss_manager.append(batch_vf_loss_manager)
                    policy_loss_manager.append(batch_loss_policy_manager)
                else:
                    batch_loss_worker, batch_policy_loss_worker, batch_vf_loss_worker, \
                    batch_entropy_worker = model.run_sgd_minibatch(
                        permutation[batch_index] * model.per_device_batch_size,
                        full_trace, self.file_writer)

                loss_worker.append(batch_loss_worker)
                policy_loss_worker.append(batch_policy_loss_worker)
                vf_loss_worker.append(batch_vf_loss_worker)
                entropy_worker.append(batch_entropy_worker)
                batch_index += 1
            if self.ES == False:
                loss_manager = np.mean(loss_manager)
                vf_loss_manager = np.mean(vf_loss_manager)
                policy_loss_manager = np.mean(policy_loss_manager)
            loss_worker = np.mean(loss_worker)
            policy_loss_worker = np.mean(policy_loss_worker)
            vf_loss_worker= np.mean(vf_loss_worker)
            entropy_worker = np.mean(entropy_worker)
            sgd_end = time.time()
            if self.ES:
                print(
                    "{:>15}{:15.5e}{:15.5e}{:15.5e}{:15.5e}".format(
                        i, loss_worker, policy_loss_worker, vf_loss_worker, entropy_worker))
            else:
                print(
                    "{:>15}{:15.5e}{:15.5e}{:15.5e}{:15.5e}{:15.5e}{:15.5e}{:15.5e}".format(
                        i, loss_manager, vf_loss_manager, policy_loss_manager, loss_worker, policy_loss_worker,
                        vf_loss_worker, entropy_worker))

            values = []
            if i == config["num_sgd_iter"] - 1:
                metric_prefix = "HRL/sgd/final_iter/"
                liste_values = [tf.Summary.Value(
                        tag=metric_prefix + "mean_entropy",
                        simple_value=entropy_worker)]
                if self.ES == False:
                    liste_values += [tf.Summary.Value(
                        tag=metric_prefix + "mean_loss_manager",
                        simple_value=policy_loss_manager),
                        tf.Summary.Value(
                            tag=metric_prefix + "mean_VF_loss_manager",
                            simple_value=vf_loss_manager)
                    ]
                liste_values += [tf.Summary.Value(
                        tag=metric_prefix + "mean_loss_worker",
                        simple_value=policy_loss_worker),
                    tf.Summary.Value(
                        tag=metric_prefix + "mean_VF_loss_worker",
                        simple_value=vf_loss_worker)]
                values.extend(liste_values)
                if self.file_writer:
                    sgd_stats = tf.Summary(value=values)
                    self.file_writer.add_summary(sgd_stats, self.global_step)
                    weights_loss_manager = model.get_weights_manager_loss()

                    for key, variable in weights_loss_manager.items():
                        log_histogram(self.file_writer, key, variable, self.global_step)

                    weights_loss_worker = model.get_weights_worker_loss()
                    for key, variable in weights_loss_worker.items():
                        log_histogram(self.file_writer, key, variable, self.global_step)


            self.global_step += 1
            sgd_time += sgd_end - sgd_start


        info = {
            "rollouts_time": rollouts_time,
            "shuffle_time": shuffle_time,
            "load_time": load_time,
            "sgd_time": sgd_time,
            "sample_throughput": len(samples["obs"]) / sgd_time
        }

        FilterManager.synchronize(
            self.local_evaluator.filters, self.remote_agents)
        res = self._fetch_metrics_from_remote_evaluators()
        res = res._replace(info=info)

        return res

    def _fetch_metrics_from_remote_evaluators(self):
        episode_rewards = []
        episode_lengths = []
        episode_rewards_agents = []

        metric_lists = [a.get_completed_rollout_metrics.remote()
                        for a in self.remote_agents]

        for metrics in metric_lists:
            episode_rewards_agents_local = []
            for episode in ray.get(metrics):
                episode_lengths.append(episode.episode_length)
                reward = episode.episode_reward
                episode_rewards.append(reward)
                episode_rewards_agents_local.append(reward)
            episode_rewards_agents.append(
                np.mean(episode_rewards_agents_local) if episode_rewards_agents_local else float('nan'))


        if self.ES:
            weights_manager_outputs = self.local_evaluator.get_weights_manager_loss()
            denominator = len(self.remote_agents) * self.config["noise_stdev"]
            for i in range(len(self.remote_agents)):
                noise = self.noise_table[i]
                for key, variable in weights_manager_outputs.items():
                    weights_manager_outputs[key] += self.config["alpha"] * (1 / denominator) * noise[key] * \
                                                    episode_rewards_agents[i]

            self.local_evaluator.set_weights_manager_loss(weights_manager_outputs)


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
            agent_state]
        pickle.dump(extra_data, open(checkpoint_path + ".extra_data", "wb"))
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.saver.restore(self.local_evaluator.sess, checkpoint_path)
        extra_data = pickle.load(open(checkpoint_path + ".extra_data", "rb"))
        self.local_evaluator.restore(extra_data[0])
        self.global_step = extra_data[1]
        ray.get([
            a.restore.remote(o)
                for (a, o) in zip(self.remote_agents, extra_data[3])])

    def compute_action(self, observation):
        observation = self.local_evaluator.obs_filter(
            observation, update=False)
        return self.local_evaluator.common_policy.compute(observation)[0]
