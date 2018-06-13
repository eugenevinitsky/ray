from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.signal
from ray.rllib.optimizers import SampleBatch


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def process_rollout(rollout, reward_filter, gamma, lambda_=1.0, use_gae=True, ADB=False):
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
        vpred_t = np.stack(rollout.data["vf_preds"] + [np.array(rollout.last_r)]).squeeze()
        delta_t = traj["rewards"] + gamma * vpred_t[1:] - vpred_t[:-1]
        advantage = discount(delta_t, gamma * lambda_)

        if ADB:
            Q_function = np.transpose(np.array(rollout.Q_function))
            Q_pred_t = np.vstack((Q_function, Q_function[-1]))
            delta_t_multi = traj["rewards"].reshape(-1, 1) + gamma * Q_pred_t[1:] - Q_pred_t[:-1]
            traj["advantages"] = discount(delta_t_multi, gamma * lambda_)
        else:
            traj["advantages"] = advantage.copy()

        #traj["value_targets"] = advantage + traj["vf_preds"]
        traj["value_targets"] = discount(traj["rewards"], gamma)

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






def dcos(a,b):
    norme_a = np.multiply(a,a).sum(axis=1)
    norme_b  = np.multiply(b,b).sum(axis=1)
    return np.multiply(a,b).sum(axis=1) / (1e-11 + np.sqrt(norme_a * norme_b) )



def compute_internal_rewards(c, liste_s, liste_g):
    shifted_s = []
    shifted_g = []
    for i in range(1, c+1):
        padding_s = np.zeros(shape=(i, liste_s.shape[1]))
        padding_g = np.zeros(shape=(i, liste_g.shape[1]))

        shifted_s.append(np.append(padding_s, liste_s[:-i], axis=0))
        shifted_g.append(np.append(padding_g, liste_g[:-i], axis=0))



    internal_rewards = 0
    for i in range(c):
        internal_rewards += dcos(liste_s- shifted_s[i], shifted_g[i])

    return internal_rewards / c



def process_rollout_Feudal(rollout, reward_filter, gamma, gamma_internal, lambda_=1.0, use_gae=True, tradeoff=0.5, c=10):
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
    trajsize = len(rollout["actions"])
    for key in rollout.data:
        traj[key] = np.stack(rollout.data[key])
    for key in rollout.data_feudal:
        traj[key] = np.stack(rollout.data_feudal[key])

    assert "manager_vf_preds" in rollout.data_feudal and "worker_vf_preds" in rollout.data_feudal, "Values not found!"

    manager_vpred_t = np.append(traj["manager_vf_preds"], np.array([traj["manager_vf_preds"][0]]))
    manager_delta_t = traj["rewards"] + gamma * manager_vpred_t[1:] - manager_vpred_t[:-1]
    # This formula for the advantage comes
    #  "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
    traj["advantages_manager"] = discount(manager_delta_t, gamma * lambda_)
    traj["value_targets_manager"] = traj["advantages_manager"] + traj["manager_vf_preds"]

    worker_vpred_t = np.append(traj["worker_vf_preds"], np.array([traj["worker_vf_preds"][0]]))
    internal_returns_ponctual = compute_internal_rewards(c, traj["s"], traj["g"])
    worker_delta_t = internal_returns_ponctual + gamma_internal * worker_vpred_t[1:] - worker_vpred_t[:-1]
    traj["advantages_worker"] = traj["advantages_manager"] + tradeoff * discount(worker_delta_t, gamma_internal * lambda_)
    traj["value_targets_worker"] = discount(worker_delta_t, gamma_internal * lambda_) + traj["worker_vf_preds"]

    for i in range(traj["advantages_manager"].shape[0]):
        traj["advantages_manager"][i] = reward_filter(traj["advantages_manager"][i])
        traj["advantages_worker"][i] = reward_filter(traj["advantages_worker"][i])

    diff_1 = np.append(traj["s"][c:], np.array([traj["s"][-1] for _ in range(c)]), axis=0)
    diff = diff_1 - traj["s"]
    traj["diff"] = diff.copy()

    del traj["s"]

    gsum = []
    g_dim = traj["g"].shape[1]
    for i in range(c + 1):
        constant = np.array([traj["g"][i] for _ in range(c - i)])
        zeros = np.zeros((i, g_dim))
        if i == 0:
            tensor = np.append(constant, traj["g"][i:i - c], axis=0)
        elif i == c:
            tensor = np.append(zeros, traj["g"][i:], axis=0)
        else:
            padding = np.append(zeros, constant, axis=0)
            tensor = np.append(padding, traj["g"][i:i - c], axis=0)

        gsum.append(tensor)

    traj["gsum"] = np.array(gsum).sum(axis=0)

    del traj["g"]


    assert all(val.shape[0] == trajsize for val in traj.values()), \
        "Rollout stacked incorrectly!"
    return SampleBatch(traj)