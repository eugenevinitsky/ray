from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.signal


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def process_rollout(rollout, reward_filter, gamma, lambda_=1.0, use_gae=True):
    """Given a rollout, compute its value targets and the advantage."""

    traj = {}
    trajsize = len(rollout.data["actions"])
    for key in rollout.data:
        traj[key] = np.stack(rollout.data[key])

    if use_gae:
        assert "vf_preds" in rollout.data, "Values not found!"
        vpred_t = np.stack(
            rollout.data["vf_preds"] + [np.array(rollout.last_r)]).squeeze()
        delta_t = traj["rewards"] + gamma * vpred_t[1:] - vpred_t[:-1]
        # This formula for the advantage comes
        # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
        traj["advantages"] = discount(delta_t, gamma * lambda_)
        traj["value_targets"] = traj["advantages"] + traj["vf_preds"]
    else:
        rewards_plus_v = np.stack(
            rollout.data["rewards"] + [np.array(rollout.last_r)]).squeeze()
        traj["advantages"] = discount(rewards_plus_v, gamma)[:-1]

    for i in range(traj["advantages"].shape[0]):
        traj["advantages"][i] = reward_filter(traj["advantages"][i])
    # FIXME(ev) this is an awful way to do this, may god have mercy on my soul
    # if there isn't an indicator that I need to rework my approach, this is it
    if len(traj['observations'].shape) > 2:
        traj['advantages'] = np.repeat(traj['advantages'], axis=0, repeats=traj['observations'].shape[1]).\
          reshape(traj['advantages'].shape[0], traj['observations'].shape[1])

    assert all(val.shape[0] == trajsize for val in traj.values()), \
        "Rollout stacked incorrectly!"
    return traj
