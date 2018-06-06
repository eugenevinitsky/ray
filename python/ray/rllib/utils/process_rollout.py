from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.signal
from ray.rllib.optimizers import SampleBatch


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def discount_sum(x, gamma, terminal=0.0):
    y = []
    run_sum = terminal
    for t in range(len(x)-1, -1, -1):
        run_sum = x[t] + gamma*run_sum
        y.append(run_sum)

    return np.array(y[::-1])


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

def process_rollout_Feudal(c, tradeoff_rewards, rollout, reward_filter, gamma, gamma_internal, ES, lambda_=1.0, lambda_internal= 0.97, use_gae=True):
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

    # TODO: WORK MORE CLOSELY ON ADVANTAGES AND TARGETS

    traj = {}
    trajsize = len(rollout.data["actions"])
    for key in rollout.data:
        traj[key] = np.squeeze(np.stack(rollout.data[key]))

    for key in rollout.data_feudal:
        traj[key] = np.squeeze(np.stack(rollout.data_feudal[key]))

    Q_functions = np.array(traj["vf_preds_worker"])
    Q_max = np.amax(Q_functions, axis=1)

    returns = discount_sum(traj["rewards"], gamma)
    internal_returns_ponctual = compute_internal_rewards(c, traj["s"], traj["g"])
    internal_returns = discount_sum(internal_returns_ponctual, gamma_internal)

    if ES == False:
        vpred_t_manager = np.append(traj["vf_preds_manager"], [np.array(rollout.last_r)], axis=0)
        delta_t_manager = traj["rewards"] + gamma * vpred_t_manager[1:] - vpred_t_manager[:-1]
        traj["advantages_manager"] = discount(delta_t_manager, gamma * lambda_)
        traj["value_targets_manager"] = returns
    else:
        traj["vf_preds_manager"] = returns
        traj["advantages_manager"] = returns
        traj["value_targets_manager"] = returns

    Q_pred_t = np.vstack((Q_functions, Q_functions[-1]))


    if ES == False:
        delta_t_worker = internal_returns_ponctual.reshape(-1, 1) + gamma_internal * Q_pred_t[1:] - Q_pred_t[:-1]
        traj["advantages_worker"] = traj["advantages_manager"].reshape(-1, 1)+ \
                                tradeoff_rewards * discount(delta_t_worker, gamma_internal * lambda_internal)
    else:
        delta_t_worker = traj["rewards"].reshape(-1, 1) + tradeoff_rewards * internal_returns_ponctual.reshape(-1, 1) + gamma_internal * Q_pred_t[1:] - Q_pred_t[:-1]
        traj["advantages_worker"] = discount(delta_t_worker, gamma_internal * lambda_internal)

    traj["value_targets_worker"] = internal_returns_ponctual + gamma_internal * Q_max


    for i in range(traj["advantages_worker"].shape[0]):
        traj["advantages_manager"][i] = reward_filter(traj["advantages_manager"][i])
        traj["advantages_worker"][i] = reward_filter(traj["advantages_worker"][i])

    traj["advantages_manager"] = traj["advantages_manager"].copy()
    traj["advantages_worker"] = traj["advantages_worker"].copy()

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

    assert all(val.shape[0] == trajsize for val in traj.values()), \
        "Rollout stacked incorrectly!"
    return SampleBatch(traj)



def process_rollout(rollout, reward_filter, gamma, ADB, lambda_=1.0, use_gae=True):
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

    returns = discount_sum(rollout.data["rewards"], gamma)
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
            vpred_t = np.stack(
                    rollout.data["vf_preds"] + [np.array(rollout.last_r)]).squeeze()
            delta_t = traj["rewards"] + gamma * vpred_t[1:] - vpred_t[:-1]
            traj["advantages"] = discount(delta_t, gamma * lambda_)


        traj["value_targets"] = returns

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
