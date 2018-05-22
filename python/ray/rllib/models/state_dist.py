from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from scipy.special import iv

class StateDistribution(object):
    """The policy action distribution of an agent.

    Args:
      inputs (Tensor): The input vector to compute samples from.
    """

    def __init__(self, inputs, kappa, obs_dim):
        self.inputs = inputs
        self.kappa = kappa
        self.obs_dim = obs_dim

    def logp(self, x):
        """The log-likelihood of the action distribution."""
        raise NotImplementedError

    def entropy(self):
        """The entroy of the action distribution."""
        raise NotImplementedError

    def sample(self):
        """Draw a sample from the action distribution."""
        raise NotImplementedError


class von_Mises_Fisher(StateDistribution):
    """Action distribution where each vector element is a gaussian.

    The first half of the input vector defines the gaussian means, and the
    second half the gaussian standard deviations.
    """

    def __init__(self, inputs, kappa, obs_dim):
        StateDistribution.__init__(self, inputs, kappa, obs_dim)
        self.mean = inputs
        self.C_p = kappa**(obs_dim/2) / ((2 * np.pi)**(obs_dim/2) * iv(obs_dim/2 - 1, kappa))

    def logp(self, x):
        dot_product = tf.reduce_sum(tf.multiply(x, self.mean), axis=1)
        return tf.cast(tf.log(self.C_p), tf.float32) + self.kappa * dot_product

    def sample(self):
        return np.random.vonmises(self.mean, self.kappa)

