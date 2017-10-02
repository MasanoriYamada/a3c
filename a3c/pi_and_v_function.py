#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F


class A3CFFSoftmaxCCFFF(chainer.Chain):
    """polocy and v network"""

    def __init__(self, n_action, thread_id):
        self.logger = logging.getLogger(__name__)
        self.n_action = n_action
        self.thread_id = thread_id

        super(A3CFFSoftmaxCCFFF, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D()
            self.conv2 = L.Convolution2D()
            self.fc1 = L.Linear(2592, 256)
            self.fc2 = L.Linear(256, self.n_action)  # pi
            self.fc3 = L.Linear(256, 1)  # v

    def get_pi_and_v(self, state):
        h1 = F.relu(self.conv1(state))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.fc1(h2))
        pi = F.softmax(self.fc2(h3), axis=1)  # action axis
        v = self.fc3(h3)
        self.logger.debug('pi: {}'.format(pi.data))
        self.logger.debug('V: {}'.format(v.data))
        return pi, v

class A3CFFSoftmaxFFF(chainer.Chain):
    """polocy and v network"""

    def __init__(self, obs_space, n_action):
        #self.logger = logging.getLogger(__name__)
        self.n_action = n_action
        self.obs_space = obs_space

        super(A3CFFSoftmaxFFF, self).__init__()
        with self.init_scope():
            self.fc1 = L.Linear(self.obs_space[0], 256)
            self.fc2 = L.Linear(256, self.n_action)  # pi
            self.fc3 = L.Linear(256, 1)  # v

    def get_pi_and_v(self, state):
        h1 = F.relu(self.fc1(state))
        pi = F.softmax(self.fc2(h1), axis=1)  # action axis
        v = self.fc3(h1)
        #self.logger.debug('pi: {}'.format(pi.data))
        #self.logger.debug('V: {}'.format(v.data))
        return pi, v

    def reset_state(self):
        pass
    # to avoid error when logger pickle and deepcopy
    def __getstate__(self):
        d = dict(self.__dict__)
        #del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)  # I *think* this is a safe way to do it

def get_entropy(pi):
    entropy = -F.sum(pi * F.log(pi), axis=1)
    return entropy

def sampled_actions_log_probs(pi, a_index):
    return F.select_item(
        F.log(pi),
        chainer.Variable(np.asarray(a_index, dtype=np.int32)))

def sampled_action(pi):
    prob = pi.data
    a_index = []  # batch

    # Subtract a tiny value from probabilities in order to avoid
    # "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial becase sum prob = 1
    prob = prob - np.finfo(np.float32).epsneg

    for i in range(prob.shape[0]):  # batch loop
        histogram = np.random.multinomial(n=1, pvals=prob[i])
        a_index.append(int(np.nonzero(histogram)[0]))

    return a_index
