#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import serializers
import chainer
import copy
import os
import logging
import numpy as np

import chainer.functions as F

from a3c.pi_and_v_function import A3CFFSoftmaxFFF
from a3c.pi_and_v_function import get_entropy
from a3c.pi_and_v_function import sampled_actions_log_probs
from a3c.pi_and_v_function import sampled_action

from a3c.constants import GAMMA
from a3c.constants import T_MAX
from a3c.constants import CLIP_REWARD_IS
from a3c.constants import ENTROPY_BETA
from a3c.constants import PI_LOSS_COEFF
from a3c.constants import V_LOSS_COEFF

def copy_param(target_link, source_link):
    """Copy parameters of a link to another link.
    """
    target_params = dict(target_link.namedparams())
    for param_name, param in source_link.namedparams():
        target_params[param_name].data[:] = param.data


def copy_grad(target_link, source_link):
    """Copy gradients of a link to another link.
    """
    target_params = dict(target_link.namedparams())
    for param_name, param in source_link.namedparams():
        target_params[param_name].grad[:] = param.grad


class Agent_a3c(object):
    def __init__(self, model, phi, optimizer):
        self.logger = logging.getLogger(__name__)
        self.shared_model = model
        #self.model = copy.deepcopy(self.shared_model)
        self.model = A3CFFSoftmaxFFF(model.obs_space, model.n_action)
        self.phi = phi
        self.clip_reward = CLIP_REWARD_IS
        self.t_max = T_MAX
        self.optimizer = optimizer

        # for calcling R = r + gamma * R, log(pi)
        self.t = 0
        self.t_start = 0
        self.past_action_log_props = {}
        self.past_rewards = {}
        self.past_Vs = {}  # Variable
        self.past_entropy = {}  # Variable


    def sync_parameters(self):
        copy_param(target_link=self.model, source_link=self.shared_model)

    def load_model(self, model_filename):
        serializers.load_hdf5(model_filename, self.model)
        copy_param(target_link=self.model, source_link=self.shared_model)
        opt_filename = model_filename + '.opt'
        if os.path.exists(opt_filename):
            print('WARNING: {0} was not found, so loaded only a model'.format(
                opt_filename))
            serializers.load_hdf5(model_filename + '.opt', self.optimizer)

    def save_model(self, model_filename):
        serializers.save_hdf5(model_filename, self.model)
        serializers.save_hdf5(model_filename + '.opt', self.optimizer)

    def act_and_train(self, state, reward, is_state_terminal):
        """agent main part
        update V and pi network
        retrun action

        R = r + gannma * R
        loss pi = log(pi)*(R-V) depend only pi
        loss V = 0.5*(R-V)**2 depend only V

        """

        if self.clip_reward:
            reward = np.clip(reward, -1, 1)
        if not is_state_terminal:
            # make batch dimension
            statevar = chainer.Variable(np.expand_dims(self.phi(state), axis=0))
        self.past_rewards[self.t-1] = reward

        if (is_state_terminal and self.t_start < self.t) or self.t - self.t_start == self.t_max:
            # get total reaward R(t)
            if is_state_terminal:
                R = 0
            else:
                _, V_out = self.model.get_pi_and_v(statevar)
                R = float(V_out.data) # R(t)

            # calc pi and V loss
            pi_loss = 0.
            V_loss = 0.
            entropy_loss = 0.
            for t in reversed(range(self.t_start, self.t)):  # t-1, t-2, ..., t_start
                self.logger.debug('local t:{}'.format(t))
                self.logger.debug('past_rewards: {}'.format(self.past_rewards ))
                self.logger.debug('past_Vs: {}'.format(self.past_Vs ))
                R = self.past_rewards[t] + GAMMA * R  # R(t-1) = r(t-1) + gamma*R(t)
                V = self.past_Vs[t]
                A = R - V

                pi_loss += self.past_action_log_props[t] * float(A.data)
                V_loss += 0.5 * (V-R)**2
                entropy_loss += self.past_entropy[t]

            # update V and pi network
            total_loss = - PI_LOSS_COEFF*pi_loss + V_LOSS_COEFF* F.reshape(V_loss, pi_loss.data.shape) - ENTROPY_BETA*entropy_loss
            self.logger.debug('pi loss: {}'.format(pi_loss.data))
            self.logger.debug('V loss: {}'.format(V_loss.data))
            self.logger.debug('entropy loss: {}'.format(entropy_loss.data))
            self.model.zerograds()
            total_loss.backward()
            self.shared_model.zerograds()
            copy_grad(target_link=self.shared_model, source_link=self.model)
            # update shared model
            self.optimizer.update()
            self.sync_parameters()
            # if use LSTM unchain because is_state_terminal

            # reset
            self.past_action_log_props = {}
            self.past_rewards = {}
            self.past_Vs = {}
            self.past_entropy = {}
            self.t = self.t_start

        # store past state
        if not is_state_terminal:
            pi_out, V_out = self.model.get_pi_and_v(statevar)
            a = sampled_action(pi_out)
            self.past_action_log_props[self.t] =sampled_actions_log_probs(pi_out, a)
            self.past_entropy[self.t] = get_entropy(pi_out)
            self.past_Vs[self.t] = V_out
            self.t += 1
            return a[0]

        else:
            self.model.reset_state()
            return None



