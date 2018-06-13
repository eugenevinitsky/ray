# Code in this file is copied and adapted from
# https://github.com/openai/evolution-strategies-starter.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import numpy as np
import os
import pickle
import time

import ray
from ray.rllib import agent

from ray.rllib.es import optimizers
from ray.rllib.es import policies
from ray.rllib.es import tabular_logger as tlogger
from ray.rllib.es import utils


Result = namedtuple("Result", [
    "noise_indices_managers", "noise_indices_worker", "noisy_returns", "noisy_internal_returns",
    "sign_noisy_returns", "sign_noisy_internal_returns", "noisy_lengths", "eval_returns", "eval_lengths"
])



DEFAULT_CONFIG = dict(
    tradeoff_coeff= 0.5,
    c=10,
    g_dim=16,
    k=16,
    z_dimension=256,
    hidden_goal_designer={"hiddens": [64, 64], "activation":"tanh"},
    l2_coeff=0.005,
    noise_stdev=0.02,
    episodes_per_batch=1000,
    timesteps_per_batch=10000,
    eval_prob=0.003,
    return_proc_mode="centered_rank",
    num_workers=10,
    stepsize=0.01,
    observation_filter="MeanStdFilter",
    noise_size=250000000,
    env_config={})


@ray.remote
def create_shared_noise(count):
    """Create a large array of noise to be shared by all workers."""
    seed = 123
    noise = np.random.RandomState(seed).randn(count).astype(np.float32)
    return noise


class SharedNoiseTable(object):
    def __init__(self, noise):
        self.noise = noise
        assert self.noise.dtype == np.float32

    def get(self, i, dim):
        return self.noise[i:i + dim]

    def sample_index(self, dim):
        return np.random.randint(0, len(self.noise) - dim + 1)


@ray.remote
class Worker(object):
    def __init__(self, registry, config, policy_params, env_creator, noise,
                 min_task_runtime=0.2):
        self.min_task_runtime = min_task_runtime
        self.config = config
        self.policy_params = policy_params
        self.noise = SharedNoiseTable(noise)

        self.env = env_creator(config["env_config"])
        from ray.rllib import models
        self.preprocessor = models.ModelCatalog.get_preprocessor(
            registry, self.env)

        self.sess = utils.make_session(single_threaded=True)
        self.policy = policies.HRLPolicy(config,
            registry, self.sess, self.env.action_space, self.preprocessor, **policy_params)


    def rollout(self, timestep_limit, add_noise=True):
        rollout_rewards, rollout_internal_rewards, rollout_length = policies.rollout(
            self.config, self.policy, self.env, timestep_limit=timestep_limit, add_noise=add_noise)
        return rollout_rewards, rollout_internal_rewards, rollout_length

    def do_rollouts(self, params_manager, params_worker, timestep_limit=None):
        # Set the network weights.
        self.policy.set_weights_manager(params_manager)
        self.policy.set_weights_worker(params_worker)

        noise_indices_manager, noise_indices_worker, returns, internal_returns, \
        sign_returns, sign_internal_returns, lengths = [], [], [], [], [], [], []
        eval_returns, eval_lengths = [], []

        # Perform some rollouts with noise.
        task_tstart = time.time()
        while (len(noise_indices_manager) == 0 or
               time.time() - task_tstart < self.min_task_runtime):

            if np.random.uniform() < self.config["eval_prob"]:
                # Do an evaluation run with no perturbation.
                self.policy.set_weights_manager(params_manager)
                self.policy.set_weights_worker(params_worker)

                rewards, internal_rewards, length = self.rollout(timestep_limit, add_noise=False)
                eval_returns.append(rewards.sum())
                eval_lengths.append(length)
            else:
                # Do a regular run with parameter perturbations.
                noise_index_manager = self.noise.sample_index(self.policy.num_params_manager)
                noise_index_worker = self.noise.sample_index(self.policy.num_params_worker)

                perturbation_manager = self.config["noise_stdev"] * self.noise.get(
                    noise_index_manager, self.policy.num_params_manager)
                perturbation_worker = self.config["noise_stdev"] * self.noise.get(
                    noise_index_worker, self.policy.num_params_worker)

                self.policy.set_weights_manager(params_manager + perturbation_manager)
                self.policy.set_weights_worker(params_worker + perturbation_worker)
                rewards_pos, internal_rewards_pos, lengths_pos = self.rollout(timestep_limit)

                self.policy.set_weights_manager(params_manager - perturbation_manager)
                self.policy.set_weights_worker(params_worker - perturbation_worker)
                rewards_neg, internal_rewards_neg, lengths_neg = self.rollout(timestep_limit)

                noise_indices_manager.append(noise_index_manager)
                noise_indices_worker.append(noise_index_worker)
                returns.append([rewards_pos.sum(), rewards_neg.sum()])
                internal_returns.append([internal_rewards_pos.sum(), internal_rewards_neg.sum()])

                sign_returns.append(
                    [np.sign(rewards_pos).sum(), np.sign(rewards_neg).sum()])
                sign_internal_returns.append(
                    [np.sign(internal_rewards_pos).sum(), np.sign(internal_rewards_neg).sum()])
                lengths.append([lengths_pos, lengths_neg])

        return Result(
            noise_indices_managers=noise_indices_manager,
            noise_indices_worker=noise_indices_worker,
            noisy_returns=returns,
            noisy_internal_returns=internal_returns,
            sign_noisy_returns=sign_returns,
            sign_noisy_internal_returns=sign_internal_returns,
            noisy_lengths=lengths,
            eval_returns=eval_returns,
            eval_lengths=eval_lengths)


class ESAgent(agent.Agent):
    _agent_name = "ES"
    _default_config = DEFAULT_CONFIG
    _allow_unknown_subkeys = ["env_config"]

    def _init(self):
        policy_params = {
            "action_noise_std": 0.01
        }

        env = self.env_creator(self.config["env_config"])
        from ray.rllib import models
        preprocessor = models.ModelCatalog.get_preprocessor(
            self.registry, env)

        self.sess = utils.make_session(single_threaded=False)
        self.policy = policies.HRLPolicy(
            self.config, self.registry, self.sess, env.action_space, preprocessor, **policy_params)
        self.optimizer_manager = optimizers.Adam(self.policy.get_weights_manager(), self.policy.num_params_manager, self.config["stepsize"])
        self.optimizer_worker = optimizers.Adam(self.policy.get_weights_worker(), self.policy.num_params_worker,
                                                 self.config["stepsize"])

        # Create the shared noise table.
        print("Creating shared noise table.")
        noise_id = create_shared_noise.remote(self.config["noise_size"])
        self.noise = SharedNoiseTable(ray.get(noise_id))

        # Create the actors.
        print("Creating actors.")
        self.workers = [
            Worker.remote(
                self.registry, self.config, policy_params, self.env_creator,
                noise_id)
            for _ in range(self.config["num_workers"])]

        self.episodes_so_far = 0
        self.timesteps_so_far = 0
        self.tstart = time.time()

    def _collect_results(self, theta_id_manager, theta_id_worker, min_episodes, min_timesteps):
        num_episodes, num_timesteps = 0, 0
        results = []
        while num_episodes < min_episodes or num_timesteps < min_timesteps:
            print(
                "Collected {} episodes {} timesteps so far this iter".format(
                    num_episodes, num_timesteps))
            rollout_ids = [worker.do_rollouts.remote(theta_id_manager, theta_id_worker)
                           for worker in self.workers]
            # Get the results of the rollouts.
            for result in ray.get(rollout_ids):
                results.append(result)
                # Update the number of episodes and the number of timesteps
                # keeping in mind that result.noisy_lengths is a list of lists,
                # where the inner lists have length 2.
                num_episodes += sum([len(pair) for pair
                                     in result.noisy_lengths])
                num_timesteps += sum([sum(pair) for pair
                                      in result.noisy_lengths])
        return results, num_episodes, num_timesteps

    def _train(self):
        config = self.config

        step_tstart = time.time()
        theta_manager = self.policy.get_weights_manager()
        theta_worker = self.policy.get_weights_worker()
        assert theta_manager.dtype == np.float32 and theta_worker.dtype == np.float32

        # Put the current policy weights in the object store.
        theta_id_manager = ray.put(theta_manager)
        theta_id_worker = ray.put(theta_worker)
        # Use the actors to do rollouts, note that we pass in the ID of the
        # policy weights.
        results, num_episodes, num_timesteps = self._collect_results(
            theta_id_manager, theta_id_worker,
            config["episodes_per_batch"],
            config["timesteps_per_batch"])

        all_noise_indices_manager = []
        all_noise_indices_worker = []
        all_training_returns = []
        all_training_internal_returns = []
        all_training_lengths = []
        all_eval_returns = []
        all_eval_lengths = []

        # Loop over the results.
        for result in results:
            all_eval_returns += result.eval_returns
            all_eval_lengths += result.eval_lengths

            all_noise_indices_manager += result.noise_indices_managers
            all_noise_indices_worker += result.noise_indices_worker
            all_training_returns += result.noisy_returns
            all_training_internal_returns += result.noisy_internal_returns
            all_training_lengths += result.noisy_lengths

        assert len(all_eval_returns) == len(all_eval_lengths)
        assert (len(all_noise_indices_manager) == len(all_noise_indices_worker) == len(all_training_returns) ==
                len(all_training_internal_returns) == len(all_training_lengths))

        self.episodes_so_far += num_episodes
        self.timesteps_so_far += num_timesteps

        # Assemble the results.
        eval_returns = np.array(all_eval_returns)
        eval_lengths = np.array(all_eval_lengths)
        noise_indices_manager = np.array(all_noise_indices_manager)
        noise_indices_worker = np.array(all_noise_indices_worker)
        noisy_returns = np.array(all_training_returns)
        noisy_internal_returns = np.array(all_training_internal_returns)
        noisy_lengths = np.array(all_training_lengths)

        # Process the returns.
        if config["return_proc_mode"] == "centered_rank":
            proc_noisy_returns = utils.compute_centered_ranks(noisy_returns)
            proc_noisy_internal_returns = utils.compute_centered_ranks(noisy_internal_returns)
            tradeoff_returns = proc_noisy_returns + config["tradeoff_coeff"] * proc_noisy_internal_returns
        else:
            raise NotImplementedError(config["return_proc_mode"])

        # Compute and take a step.

        """Manager"""
        g_manager, count = utils.batched_weighted_sum(
            proc_noisy_returns[:, 0] - proc_noisy_returns[:, 1],
            (self.noise.get(index, self.policy.num_params_manager)
             for index in noise_indices_manager),
            batch_size=500)
        g_manager /= noisy_returns.size
        # Compute the new weights theta.
        theta_manager, update_ratio_manager = self.optimizer_manager.update(
            -g_manager + config["l2_coeff"] * theta_manager)


        """Worker"""
        g_worker, count_worker = utils.batched_weighted_sum(
            tradeoff_returns[:, 0] - tradeoff_returns[:, 1],
            (self.noise.get(index, self.policy.num_params_worker)
             for index in noise_indices_worker),
            batch_size=500)
        g_worker /= noisy_returns.size
        assert (
                g_manager.shape == (self.policy.num_params_manager,) and
                g_worker.shape == (self.policy.num_params_worker,) and
                g_manager.dtype == np.float32 and
                g_worker.dtype == np.float32 and
                count == len(noise_indices_manager) and
                count_worker == len(noise_indices_worker))
        # Compute the new weights theta.
        theta_worker, update_ratio_worker = self.optimizer_worker.update(
            -g_worker + config["l2_coeff"] * theta_worker)


        # Set the new weights in the local copy of the policy.
        self.policy.set_weights_manager(theta_manager)
        self.policy.set_weights_worker(theta_worker)

        step_tend = time.time()
        tlogger.record_tabular("EvalEpRewMean", eval_returns.mean())
        tlogger.record_tabular("EvalEpRewStd", eval_returns.std())
        tlogger.record_tabular("EvalEpLenMean", eval_lengths.mean())

        tlogger.record_tabular("EpRewMean", noisy_returns.mean())
        tlogger.record_tabular("EpRewStd", noisy_returns.std())
        tlogger.record_tabular("EpLenMean", noisy_lengths.mean())

        tlogger.record_tabular("Norm_manager", float(np.square(theta_manager).sum()))
        tlogger.record_tabular("GradNorm_manager", float(np.square(g_manager).sum()))
        tlogger.record_tabular("UpdateRatio_manager", float(update_ratio_manager))

        tlogger.record_tabular("Norm_worker", float(np.square(theta_worker).sum()))
        tlogger.record_tabular("GradNorm_worker", float(np.square(g_worker).sum()))
        tlogger.record_tabular("UpdateRatio_worker", float(update_ratio_worker))

        tlogger.record_tabular("EpisodesThisIter", noisy_lengths.size)
        tlogger.record_tabular("EpisodesSoFar", self.episodes_so_far)
        tlogger.record_tabular("TimestepsThisIter", noisy_lengths.sum())
        tlogger.record_tabular("TimestepsSoFar", self.timesteps_so_far)

        tlogger.record_tabular("TimeElapsedThisIter", step_tend - step_tstart)
        tlogger.record_tabular("TimeElapsed", step_tend - self.tstart)
        tlogger.dump_tabular()

        info = {
            "weights_norm_manager": np.square(theta_manager).sum(),
            "weights_norm_worker": np.square(theta_worker).sum(),
            "grad_norm_manager": np.square(g_manager).sum(),
            "grad_norm_worker": np.square(g_worker).sum(),
            "update_ratio_manager": update_ratio_manager,
            "update_ratio_worker": update_ratio_worker,
            "episodes_this_iter": noisy_lengths.size,
            "episodes_so_far": self.episodes_so_far,
            "timesteps_this_iter": noisy_lengths.sum(),
            "timesteps_so_far": self.timesteps_so_far,
            "time_elapsed_this_iter": step_tend - step_tstart,
            "time_elapsed": step_tend - self.tstart
        }

        result = ray.tune.result.TrainingResult(
            episode_reward_mean=eval_returns.mean(),
            episode_len_mean=eval_lengths.mean(),
            timesteps_this_iter=noisy_lengths.sum(),
            info=info)

        return result

    def _stop(self):
        # workaround for https://github.com/ray-project/ray/issues/1516
        for w in self.workers:
            w.__ray_terminate__.remote(w._ray_actor_id.id())

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(
            checkpoint_dir, "checkpoint-{}".format(self.iteration))
        weights_manager = self.policy.get_weights_manager()
        weights_worker = self.policy.get_weights_worker()
        objects = [
            weights_manager, weights_worker,
            self.episodes_so_far,
            self.timesteps_so_far]
        pickle.dump(objects, open(checkpoint_path, "wb"))
        return checkpoint_path

    def _restore(self, checkpoint_path):
        objects = pickle.load(open(checkpoint_path, "rb"))
        self.policy.set_weights(objects[0])
        self.episodes_so_far = objects[1]
        self.timesteps_so_far = objects[2]

    def compute_action(self, observation):
        return self.policy.compute(observation, update=False)[0]
