from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.signal
from ray.rllib.optimizers import SampleBatch


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def process_rollout(rollout, reward_filter, gamma, lambda_=1.0, use_gae=True, ADB=ADB):
    """Given a rollout, compute its value targets and the advantage.
    Args:
        rollout (PartialRollout): Partial Rollout Object
        reward_filter (Filter): Filter for processing advantanges
        gamma (float): Parameter for GAE
        lambda_ (float): Parameter for GAE
        use_gae (bool): Using Generalized Advantage Estamation
    Returns:
        SampleBatch (SampleBatch): Object with experience from rollout and
            processed rewards."""

    traj = {}
    trajsize = len(rollout.data["actions"])
    for key in rollout.data:
        traj[key] = np.stack(rollout.data[key])

    if use_gae:
        if ADB:
            Q_function = np.transpose(np.array(rollout.Q_function))
            Q_pred_t = np.vstack((Q_function, Q_function[-1]))
            delta_t_multi = traj["rewards"].reshape(-1, 1) + gamma * Q_pred_t[1:] - Q_pred_t[:-1]
            traj["advantages"] = discount(delta_t_multi, gamma * lambda_)
        else:
            vpred_t = np.stack(rollout.data["vf_preds"] + [np.array(rollout.last_r)]).squeeze()
            delta_t = traj["rewards"] + gamma * vpred_t[1:] - vpred_t[:-1]
            traj["advantages"] = discount(delta_t, gamma * lambda_)
            #traj["advantages"] = returns - vpred_t[:-1]

        traj["value_targets"] = traj["advantages"] + traj["vf_preds"]
        #traj["value_targets"] = traj["rewards"] + gamma * traj["vf_preds"]
        #traj["value_targets"] = returns

    else:
        rewards_plus_v = np.stack(
            rollout.data["rewards"] + [np.array(rollout.last_r)]).squeeze()
        traj["advantages"] = discount(rewards_plus_v, gamma)[:-1]

    for i in range(traj["advantages"].shape[0]):
        traj["advantages"][i] = reward_filter(traj["advantages"][i])

    traj["advantages"] = traj["advantages"].copy()

    assert all(val.shape[0] == trajsize for val in traj.values()), \
        "Rollout stacked incorrectly!"
    return SampleBatch(traj)