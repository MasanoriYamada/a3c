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
import chainer.optimizer

from a3c.constants import LEARNING_RATE
from a3c.constants import RMSPROP_EPS
from a3c.rmsprop_async import RMSpropAsync
from a3c.nonbias_weight_decay import NonbiasWeightDecay
from a3c.experiments import async_train
from a3c.agent_a3c import Agent_a3c
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

    agent =  Agent_a3c(shared_model, phi, opt)
    async_train(env_name, agent)

    logger.info('END')

if __name__ == '__main__':
    main()