from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import ray
from ray.rllib.agent import Agent
from ray.rllib.ddpg.ddpg_evaluator import DDPGEvaluator, RemoteDDPGEvaluator
from ray.rllib.optimizers import LocalSyncReplayOptimizer
from ray.tune.result import TrainingResult

OPTIMIZER_SHARED_CONFIGS = [
    "buffer_size", "prioritized_replay", "prioritized_replay_alpha",
    "prioritized_replay_beta", "prioritized_replay_eps", "sample_batch_size",
    "train_batch_size", "learning_starts", "clip_rewards"
]

DEFAULT_CONFIG = {
    # === Model ===
    # Hidden layer sizes of the policy networks
    'actor_hiddens': [64, 64],
    # Hidden layer sizes of the policy networks
    'critic_hiddens': [64, 64],
    # N-step Q learning
    'n_step': 1,
    # Config options to pass to the model constructor
    'model': {},
    # Discount factor for the MDP
    'gamma': 0.99,
    # Arguments to pass to the env creator
    'env_config': {},

    # === Exploration ===
    # Max num timesteps for annealing schedules. Exploration is annealed from
    # 1.0 to exploration_fraction over this number of timesteps scaled by
    # exploration_fraction
    'schedule_max_timesteps': 100000,
    # Number of env steps to optimize for before returning
    'timesteps_per_iteration': 1000,
    # Fraction of entire training period over which the exploration rate is
    # annealed
    'exploration_fraction': 0.1,
    # Final value of random action probability
    'exploration_final_eps': 0.02,
    # OU-noise scale
    'noise_scale': 0.1,
    # theta
    'exploration_theta': 0.15,
    # sigma
    'exploration_sigma': 0.2,
    # Update the target network every `target_network_update_freq` steps.
    'target_network_update_freq': 0,
    # Update the target by \tau * policy + (1-\tau) * target_policy
    'tau': 0.002,
    # Whether to start with random actions instead of noops.
    'random_starts': True,

    # === Replay buffer ===
    # Size of the replay buffer. Note that if async_updates is set, then
    # each worker will have a replay buffer of this size.
    'buffer_size': 50000,
    # If True prioritized replay buffer will be used.
    'prioritized_replay': True,
    # Alpha parameter for prioritized replay buffer.
    'prioritized_replay_alpha': 0.6,
    # Beta parameter for sampling from prioritized replay buffer.
    'prioritized_replay_beta': 0.4,
    # Epsilon to add to the TD errors when updating priorities.
    'prioritized_replay_eps': 1e-6,
    # Whether to clip rewards to [-1, 1] prior to adding to the replay buffer.
    'clip_rewards': True,

    # === Optimization ===
    # Learning rate for adam optimizer
    'actor_lr': 1e-4,
    'critic_lr': 1e-3,
    # If True, use huber loss instead of squared loss for critic network
    # Conventionally, no need to clip gradients if using a huber loss
    'use_huber': False,
    # Threshold of a huber loss
    'huber_threshold': 1.0,
    # Weights for L2 regularization
    'l2_reg': 1e-6,
    # If not None, clip gradients during optimization at this value
    'grad_norm_clipping': None,
    # How many steps of the model to sample before learning starts.
    'learning_starts': 1500,
    # Update the replay buffer with this many samples at once. Note that this
    # setting applies per-worker if num_workers > 1.
    'sample_batch_size': 1,
    # Size of a batched sampled from replay buffer for training. Note that
    # if async_updates is set, then each worker returns gradients for a
    # batch of this size.
    'train_batch_size': 256,
    # Smooth the current average reward over this many previous episodes.
    'smoothing_num_episodes': 100,

    # === Tensorflow ===
    # Arguments to pass to tensorflow
    'tf_session_args': {
        "device_count": {
            "CPU": 2
        },
        "log_device_placement": False,
        "allow_soft_placement": True,
        "gpu_options": {
            "allow_growth": True
        },
        "inter_op_parallelism_threads": 1,
        "intra_op_parallelism_threads": 1,
    },

    # === Parallelism ===
    # Number of workers for collecting samples with. This only makes sense
    # to increase if your environment is particularly slow to sample, or if
    # you're using the Async or Ape-X optimizers.
    'num_workers': 0,
    # Whether to allocate GPUs for workers (if > 0).
    'num_gpus_per_worker': 0,
    # Optimizer class to use.
    'optimizer_class': "LocalSyncReplayOptimizer",
    # Config to pass to the optimizer.
    'optimizer_config': {},
    # Whether to use a distribution of epsilons across workers for exploration.
    'per_worker_exploration': False,
    # Whether to compute priorities on workers.
    'worker_side_prioritization': False
}


class DDPGAgent(Agent):
    _agent_name = "DDPG"
    _default_config = DEFAULT_CONFIG

    def _init(self):
        self.local_evaluator = DDPGEvaluator(
            self.registry, self.env_creator, self.config)
        self.remote_evaluators = [
            RemoteDDPGEvaluator.remote(
                self.registry, self.env_creator, self.config)
            for _ in range(self.config["num_workers"])]
        self.optimizer = LocalSyncReplayOptimizer(
            self.config["optimizer"], self.local_evaluator,
            self.remote_evaluators)

    def _train(self):
        for _ in range(self.config["train_steps"]):
            self.optimizer.step()
            # update target
            if self.optimizer.num_steps_trained > 0:
                self.local_evaluator.update_target()

        # generate training result
        return self._fetch_metrics()

    def _fetch_metrics(self):
        episode_rewards = []
        episode_lengths = []
        if self.config["num_workers"] > 0:
            metric_lists = [a.get_completed_rollout_metrics.remote()
                            for a in self.remote_evaluators]
            for metrics in metric_lists:
                for episode in ray.get(metrics):
                    episode_lengths.append(episode.episode_length)
                    episode_rewards.append(episode.episode_reward)
        else:
            metrics = self.local_evaluator.get_completed_rollout_metrics()
            for episode in metrics:
                episode_lengths.append(episode.episode_length)
                episode_rewards.append(episode.episode_reward)

        avg_reward = (np.mean(episode_rewards))
        avg_length = (np.mean(episode_lengths))
        timesteps = np.sum(episode_lengths)

        result = TrainingResult(
            episode_reward_mean=avg_reward,
            episode_len_mean=avg_length,
            timesteps_this_iter=timesteps,
            info={})

        return result
