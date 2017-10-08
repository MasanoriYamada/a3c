#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import numpy as np
import gym
import logging
import logging.config
import datetime
import chainer.optimizer
from tensorboard import SummaryWriter

from a3c.constants import LEARNING_RATE
from a3c.constants import RMSPROP_EPS
from a3c.rmsprop_async import RMSpropAsync
from a3c.nonbias_weight_decay import NonbiasWeightDecay
from a3c.experiments import async_train
from a3c.pi_and_v_function import A3CFFSoftmaxFFF


def phi(obs):
    return obs.astype(np.float32)

def main():

    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    action_space = env.action_space.n
    observation_space = env.observation_space.low.shape
    # set logger
    logging.config.fileConfig('./log/log.conf')
    logger = logging.getLogger(__name__)
    logger.info('START')

    # set network model
    shared_model = A3CFFSoftmaxFFF(observation_space, action_space)
    # set optimizer
    opt = RMSpropAsync(lr=LEARNING_RATE , alpha=0.99 , eps=RMSPROP_EPS)
    opt.setup(shared_model)
    opt.add_hook(chainer.optimizer.GradientClipping(40))

    writer = SummaryWriter('results/' + datetime.datetime.now().strftime('%B%d  %H:%M:%S'))
    state = env.reset()
    state = chainer.Variable(np.expand_dims(np.array(state).astype(np.float32), axis=0))
    pi, v = shared_model.get_pi_and_v(state)
    writer.add_graph([pi, v])
    writer.close()

    async_train(env_name, shared_model, opt, phi)

    logger.info('END')

if __name__ == '__main__':
    main()