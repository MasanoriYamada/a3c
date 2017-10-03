#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import chainer
import chainer.optimizers
from nose.tools import raises, assert_true, assert_equal, assert_raises

import a3c.agent_a3c
from a3c.constants import CLIP_REWARD_IS

class Dummy_model(chainer.Chain):
    def __init__(self):
        self.obs_space = [1]
        self.n_action = 1
    def get_pi_and_v(self, state):
        v2 = chainer.Variable(np.array([[1.0]]))
        v3 = chainer.Variable(np.array([[3.0]]))
        print('pi: {}'.format(v2.data))
        print('v: {}'.format(v3.data))
        return v2, v3
    def reset_state(self):
        pass

class TestAgent_a3c(object):

    def setup(self):
        opt = chainer.optimizers.RMSprop()
        phi = lambda x: x.astype(np.float32, copy=False)
        model = Dummy_model()
        self.agent = a3c.agent_a3c.Agent_a3c(model=model, phi=phi, optimizer=opt)

    def test_R__check(self):
        dummy_s = np.array([0])
        r0 = 0.  #  terminal reward
        r1 = 1.0
        r2 = 2.0
        r3 = 3.0
        g = 0.99

        # careful clip reward = False case
        self.agent.act_and_train(dummy_s, 0, False)  #ignore first reward
        self.agent.act_and_train(dummy_s, r1, True)
        assert_equal(self.agent.R, r1 + g * 3.0)

        self.agent.act_and_train(dummy_s, 0, False)
        self.agent.act_and_train(dummy_s, r1, False)
        self.agent.act_and_train(dummy_s, r2, True)
        assert_equal(self.agent.R, r1 + g*r2 + g*g*3.0)

        self.agent.act_and_train(dummy_s, 0, False)
        self.agent.act_and_train(dummy_s, r1, False)
        self.agent.act_and_train(dummy_s, r2, False)
        self.agent.act_and_train(dummy_s, r3, True)
        assert_equal(self.agent.R, r1 + g*r2 + g*g*r3 + g*g*g*3.0)

        # assert_equal(self.agent.pi_loss.data, np.array([0.1]))
        # assert_equal(self.agent.V_loss.data, np.array([[0.9699999999999998]]))
        # assert_equal(self.agent.entropy_loss.data, np.array([0.]))