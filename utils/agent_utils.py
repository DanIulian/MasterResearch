import numpy as np
import scipy.signal
import torch.nn as nn

def to_cuda(data, use_cuda):
    input_ = data.float()
    if use_cuda:
        input_ = input_.cuda()
    return input_


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def init_cnn(module, active=True):

    if not active:
        return module

    return init(
        module, nn.init.orthogonal_,
        lambda x: nn.init.constant_(x, 0),
        nn.init.calculate_gain('relu'))


def init_fc(module, activate=True):

    if not activate:
        return module

    return init(
        module, nn.init.orthogonal_,
        lambda x: nn.init.constant_(x, 0))


def init_mlp(module, activate=True):

    if not activate:
        return  module

    return init(
        module,
        nn.init.orthogonal_,
        lambda x: nn.init.constant_(x, 0),
        np.sqrt(2))


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count
    return new_mean, new_var, new_count


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = \
            update_mean_var_count_from_moments(
                self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


