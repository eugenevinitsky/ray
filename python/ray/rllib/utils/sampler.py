from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six.moves.queue as queue
import threading
from collections import namedtuple
import numpy as np


class PartialRollout(object):
    """A piece of a complete rollout.

    We run our agent, and process its experience once it has processed enough
    steps.

    Attributes:
        data (dict): Stores rollout data. All numpy arrays other than
            `observations` and `features` will be squeezed.
        last_r (float): Value of next state. Used for bootstrapping.
    """

    fields = ["obs", "actions", "rewards", "new_obs", "dones", "features"]

    def __init__(self, extra_fields=None):
        """Initializers internals. Maintains a `last_r` field
        in support of partial rollouts, used in bootstrapping advantage
        estimation.

        Args:
            extra_fields: Optional field for object to keep track.
        """
        if extra_fields:
            self.fields.extend(extra_fields)
        self.data = {k: [] for k in self.fields}
        self.last_r = 0.0
        self.Q_function = []


    def add(self, **kwargs):
        for k, v in kwargs.items():
            self.data[k] += [v]

    def set_Q_function(self, liste):
        self.Q_function = liste

    def extend(self, other_rollout):
        """Extends internal data structure. Assumes other_rollout contains
        data that occured afterwards."""

        assert not self.is_terminal()
        assert all(k in other_rollout.fields for k in self.fields)
        for k, v in other_rollout.data.items():
            self.data[k].extend(v)
        self.last_r = other_rollout.last_r

    def is_terminal(self):
        """Check if terminal.

        Returns:
            terminal (bool): if rollout has terminated."""
        return self.data["dones"][-1]

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, item):
        self.data[key] = item

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def __iter__(self):
        return self.data.__iter__()

    def __next__(self):
        return self.data.__next__()

    def __contains__(self, x):
        return x in self.data


CompletedRollout = namedtuple("CompletedRollout",
                              ["episode_length", "episode_reward"])

class AsyncSampler(threading.Thread):
    """This class interacts with the environment and tells it what to do.

    Note that batch_size is only a unit of measure here. Batches can
    accumulate and the gradient can be calculated on up to 5 batches."""
    _async = True

    def __init__(self, env, policy, obs_filter, num_local_steps, horizon=None):
        assert getattr(
            obs_filter, "is_concurrent",
            False), ("Observation Filter must support concurrent updates.")
        threading.Thread.__init__(self)
        self.queue = queue.Queue(5)
        self.metrics_queue = queue.Queue()
        self.num_local_steps = num_local_steps
        self.horizon = horizon
        self.env = env
        self.policy = policy
        self._obs_filter = obs_filter
        self.started = False
        self.daemon = True

    def run(self):
        self.started = True
        try:
            self._run()
        except BaseException as e:
            self.queue.put(e)
            raise e

    def _run(self):
        rollout_provider = _env_runner(self.env, self.policy,
                                       self.num_local_steps, self.horizon,
                                       self._obs_filter)
        while True:
            # The timeout variable exists because apparently, if one worker
            # dies, the other workers won't die with it, unless the timeout is
            # set to some large number. This is an empirical observation.
            item = next(rollout_provider)
            if isinstance(item, CompletedRollout):
                self.metrics_queue.put(item)
            else:
                self.queue.put(item, timeout=600.0)

    def get_data(self):
        """Gets currently accumulated data.

        Returns:
            rollout (PartialRollout): trajectory data (unprocessed)
        """
        assert self.started, "Sampler never started running!"
        rollout = self.queue.get(timeout=600.0)
        if isinstance(rollout, BaseException):
            raise rollout
        while not rollout.is_terminal():
            try:
                part = self.queue.get_nowait()
                if isinstance(part, BaseException):
                    raise rollout
                rollout.extend(part)
            except queue.Empty:
                break
        return rollout

    def get_metrics(self):
        completed = []
        while True:
            try:
                completed.append(self.metrics_queue.get_nowait())
            except queue.Empty:
                break
        return completed


class SyncSampler(object):
    """This class interacts with the environment and tells it what to do.
    Note that batch_size is only a unit of measure here. Batches can
    accumulate and the gradient can be calculated on up to 5 batches.
    This class provides data on invocation, rather than on a separate
    thread."""
    async = False

    def __init__(self, env, policy, obs_filter,
                 num_local_steps, horizon=None, ADB=False):
        self.ADB = ADB
        self.num_local_steps = num_local_steps
        self.horizon = horizon
        self.env = env
        self.policy = policy
        self._obs_filter = obs_filter
        self.rollout_provider = _env_runner(
            self.env, self.policy, self.num_local_steps, self.horizon,
            self._obs_filter, self.ADB)
        self.metrics_queue = queue.Queue()

    def get_data(self):
        while True:
            item = next(self.rollout_provider)
            if isinstance(item, CompletedRollout):
                self.metrics_queue.put(item)
            else:
                return item

    def get_metrics(self):
        completed = []
        while True:
            try:
                completed.append(self.metrics_queue.get_nowait())
            except queue.Empty:
                break
        return completed

def _env_runner(env, policy, num_local_steps, horizon, obs_filter, ADB):

    last_observation = obs_filter(env.reset())
    try:
        horizon = horizon if horizon else env.spec.max_episode_steps
    except Exception:
        print("Warning, no horizon specified, assuming infinite")
    if not horizon:
        horizon = 999999
    if hasattr(policy, "get_initial_features"):
        last_features = policy.get_initial_features()
    else:
        last_features = []
    features = last_features
    length = 0
    rewards = 0
    rollout_number = 0

    while True:
        terminal_end = False
        rollout = PartialRollout(extra_fields=policy.other_output)

        for step in range(num_local_steps):
            action, pi_info = policy.compute(last_observation, *last_features)
            if policy.is_recurrent:
                features = pi_info["features"]
                del pi_info["features"]
            observation, reward, terminal, info = env.step(action)
            observation = obs_filter(observation)

            length += 1
            rewards += reward
            if length >= horizon:
                terminal = True

            # Concatenate multiagent actions
            if isinstance(action, list):
                action = np.concatenate(action, axis=0).flatten()

            # Collect the experience.
            rollout.add(obs=last_observation,
                        actions=action,
                        rewards=reward,
                        dones=terminal,
                        features=last_features,
                        new_obs=observation,
                        **pi_info)

            last_observation = observation
            last_features = features

            if terminal:
                terminal_end = True
                yield CompletedRollout(length, rewards)

                if (length >= horizon or
                        not env.metadata.get("semantics.autoreset")):
                    last_observation = obs_filter(env.reset())
                    if hasattr(policy, "get_initial_features"):
                        last_features = policy.get_initial_features()
                    else:
                        last_features = []
                    rollout_number += 1
                    length = 0
                    rewards = 0
                    break

        if not terminal_end:
            rollout.last_r = policy.value(last_observation, *last_features)

        if ADB:
            Q_functions = policy.compute_Q_fuctions(rollout.data["obs"], rollout.data["actions"])
            rollout.set_Q_function(Q_functions)

        # Once we have enough experience, yield it, and have the ThreadRunner
        # place it on a queue.
        yield rollout





class PartialRollout_Feudal(object):
    """A piece of a complete rollout.

    We run our agent, and process its experience once it has processed enough
    steps.

    Attributes:
        data (dict): Stores rollout data. All numpy arrays other than
            `observations` and `features` will be squeezed.
        last_r (float): Value of next state. Used for bootstrapping.
    """

    fields = ["obs", "actions", "rewards", "new_obs", "dones", "features"]
    fields_feudal = ["manager_vf_preds", "worker_vf_preds", "logprobs", "z_carried", "s", "g"]

    def __init__(self, extra_fields=None):
        """Initializers internals. Maintains a `last_r` field
        in support of partial rollouts, used in bootstrapping advantage
        estimation.

        Args:
            extra_fields: Optional field for object to keep track.
        """
        if extra_fields:
            self.fields.extend(extra_fields)
        self.data = {k: [] for k in self.fields}
        self.data_feudal = {k: [] for k in self.fields_feudal}
        self.last_r = 0.0

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self.data[k] += [v]
    def add_Feudal(self, **kwargs):
        for k, v in kwargs.items():
            self.data_feudal[k] += [v]

    def extend(self, other_rollout):
        """Extends internal data structure. Assumes other_rollout contains
        data that occured afterwards."""

        assert not self.is_terminal()
        assert all(k in other_rollout.fields for k in self.fields)
        for k, v in other_rollout.data.items():
            self.data[k].extend(v)
        self.last_r = other_rollout.last_r

    def is_terminal(self):
        """Check if terminal.

        Returns:
            terminal (bool): if rollout has terminated."""
        return self.data["dones"][-1]

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, item):
        self.data[key] = item

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def __iter__(self):
        return self.data.__iter__()

    def __next__(self):
        return self.data.__next__()

    def __contains__(self, x):
        return x in self.data




class SyncSampler_Feudal(object):
    """This class interacts with the environment and tells it what to do.

    Note that batch_size is only a unit of measure here. Batches can
    accumulate and the gradient can be calculated on up to 5 batches.

    This class provides data on invocation, rather than on a separate
    thread."""
    _async = False

    def __init__(self, env, policy, obs_filter, config):
        self.config = config
        self.num_local_steps = config["horizon"]
        self.horizon = config["horizon"]
        self.env = env
        self.policy = policy
        self._obs_filter = obs_filter
        self.rollout_provider = _env_runner_Feudal(self.env, self.policy,
                                            self.num_local_steps, self.horizon,
                                            self._obs_filter, self.config)
        self.metrics_queue = queue.Queue()

    def get_data(self):
        while True:
            item = next(self.rollout_provider)
            if isinstance(item, CompletedRollout):
                self.metrics_queue.put(item)
            else:
                return item

    def get_metrics(self):
        completed = []
        while True:
            try:
                completed.append(self.metrics_queue.get_nowait())
            except queue.Empty:
                break
        return completed



def _env_runner_Feudal(env, policy, num_local_steps, horizon, obs_filters, config):
    """This implements the logic of the thread runner.

    It continually runs the policy, and as long as the rollout exceeds a
    certain length, the thread runner appends the policy to the queue. Yields
    when `timestep_limit` is surpassed, environment terminates, or
    `num_local_steps` is reached.

    Args:
        env: Environment generated by env_creator
        policy: Policy used to interact with environment. Also sets fields
            to be included in `PartialRollout`
        num_local_steps: Number of steps before `PartialRollout` is yielded.
        obs_filter: Filter used to process observations.

    Yields:
        rollout (PartialRollout): Object containing state, action, reward,
            terminal condition, and other fields as dictated by `policy`.
    """
    c = config["c"]


    """Inputs are difference of two consecutive frames"""
    current_image = obs_filters(env.reset())
    prev_image = current_image.copy()
    last_observation = current_image - prev_image
    """
    last_observation = obs_filters(env.reset())
    """

    try:
        horizon = horizon if horizon else env.spec.max_episode_steps
    except Exception:
        print("Warning, no horizon specified, assuming infinite")
    if not horizon:
        horizon = 999999
    if hasattr(policy, "get_initial_features"):
        last_features = policy.get_initial_features()
    else:
        last_features = []
    features = last_features
    length = 0
    rewards = 0
    rollout_number = 0
    g_s = 0
    while True:
        terminal_end = False
        rollout = PartialRollout_Feudal(extra_fields=policy.other_output)

        for step in range(num_local_steps):
            z, s, g, vfm = policy.compute_manager(last_observation, *last_features)

            """
            if np.random.rand() < config["epsilon"]:
                g = np.random.normal(loc=0.0, scale=1.0, size=g.shape)
            """

            if step == 0:
                g_s = np.array([g])
                g_sum = g
            elif step < config["c"]:
                g_s = np.append(g_s, [g], axis=0)
                g_sum = g_s.sum(axis=0)
            else:
                g_s = np.append(g_s, [g], axis=0)
                g_sum = g_s[-(c+1):].sum(axis=0)

            action, logprobs, vfw = policy.compute_worker(z, g_sum)
            observation, reward, terminal, info = env.step(action)
            observation = obs_filters(observation)



            length += 1
            rewards += reward
            if length >= horizon:
                terminal = True

            # Concatenate multiagent actions
            if isinstance(action, list):
                action = np.concatenate(action, axis=0).flatten()

            # Collect the experience.
            rollout.add(
                obs=last_observation,
                actions=action,
                rewards=reward,
                dones=terminal,
                features=last_features,
                new_obs=observation)
            rollout.add_Feudal(manager_vf_preds=vfm,
                               worker_vf_preds=vfw,
                               logprobs=logprobs,
                               z_carried=z,
                               s=s,
                               g=g)



            prev_image = current_image.copy()
            current_image = observation.copy()
            last_observation = current_image - prev_image
            """
            last_observation = observation
            """

            last_features = features

            if terminal:
                terminal_end = True
                yield CompletedRollout(length, rewards)

                if (length >= horizon
                        or not env.metadata.get("semantics.autoreset")):


                    current_image = obs_filters(env.reset())
                    prev_image = current_image.copy()
                    last_observation = current_image - prev_image
                    """
                    last_observation = obs_filters(env.reset())
                    """
                    if hasattr(policy, "get_initial_features"):
                        last_features = policy.get_initial_features()
                    else:
                        last_features = []
                    rollout_number += 1
                    length = 0
                    rewards = 0
                    g_s = 0
                    break

        if not terminal_end:
            rollout.last_r = policy.value(last_observation, *last_features)
        # Once we have enough experience, yield it, and have the ThreadRunner
        # place it on a queue.
        yield rollout