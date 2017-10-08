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
    def __init__(self, shared_model, phi, optimizer):
        self.logger = logging.getLogger(__name__)
        self.shared_model = shared_model
        #self.model = copy.deepcopy(self.shared_model)
        self.local_model = None
        self.phi = phi
        self.clip_reward = CLIP_REWARD_IS
        self.t_max = T_MAX
        self.optimizer = optimizer
        self.thread_id = None

        # for calcling R = r + gamma * R, log(pi)
        self.t = 0
        self.t_start = 0
        self.past_action_log_props = {}
        self.past_rewards = {}
        self.past_Vs = {}  # Variable
        self.past_entropy = {}  # Variable

    def generagte_local_model(self, thread_id):
        self.local_model = A3CFFSoftmaxFFF(self.shared_model.obs_space, self.shared_model.n_action)
        self.thread_id = thread_id
        
    def sync_parameters(self):
        copy_param(target_link=self.local_model, source_link=self.shared_model)

    def load_model(self, model_filename):
        serializers.load_hdf5(model_filename, self.local_model)
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
        #if not is_state_terminal:
            # make batch dimension
        statevar = chainer.Variable(np.expand_dims(self.phi(state), axis=0))
        self.past_rewards[self.t-1] = float(reward)

        if (is_state_terminal and self.t_start < self.t) or self.t - self.t_start == self.t_max:
            assert self.t_start < self.t

            # get total reward R(t)
            if is_state_terminal:
                self.R = 0
            else:
                _, V_out = self.local_model.get_pi_and_v(statevar)
                self.R = float(V_out.data) # R(t)

            # calc pi and V loss
            self.pi_loss = 0.
            self.V_loss = 0.
            self.entropy_loss = 0.
            self.total_loss = 0.
            for t in reversed(range(self.t_start, self.t)):  # t-1, t-2, ..., t_start
                self.logger.debug('local t:{}'.format(t))
                self.logger.debug('past_rewards: {}'.format(self.past_rewards))
                self.logger.debug('past_Vs: {}'.format(self.past_Vs))
                self.logger.debug('past_entropy: {}'.format(self.past_entropy))
                self.R = self.past_rewards[t] + GAMMA * self.R  # R(t-1) = r(t-1) + gamma*R(t)
                self.V = self.past_Vs[t]
                self.A = self.R - self.V
                self.pi_loss += self.past_action_log_props[t] * float(self.A.data)
                self.V_loss += 0.5 * (self.V-self.R)**2
                self.entropy_loss += self.past_entropy[t]

                self.logger.debug('R: {}'.format(self.R))
                self.logger.debug('A: {}'.format(self.A))
                self.logger.debug('local pi loss: {}'.format(self.pi_loss.data))
                self.logger.debug('local v loss: {}'.format(self.V_loss.data))
                self.logger.debug('local entropy loss: {}'.format(self.entropy_loss.data))

            # update V and pi network
            self.total_loss = - PI_LOSS_COEFF*self.pi_loss \
                              + V_LOSS_COEFF* F.reshape(self.V_loss, self.pi_loss.data.shape) \
                              - ENTROPY_BETA*self.entropy_loss

            self.logger.debug('pi loss: {}'.format(self.pi_loss.data))
            self.logger.debug('V loss: {}'.format(self.V_loss.data))
            #  self.logger.debug('entropy loss: {}'.format(self.entropy_loss.data))
            self.logger.debug('total loss: {}'.format(self.total_loss.data))
            self.logger.debug('model update')
            self.local_model.zerograds()
            self.total_loss.backward()
            self.shared_model.zerograds()
            copy_grad(target_link=self.shared_model, source_link=self.local_model)
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

            # for debug
            self.shared_pi_out, self.shared_V_out = self.shared_model.get_pi_and_v(statevar)

        # store past state
        if not is_state_terminal:
            pi_out, V_out = self.local_model.get_pi_and_v(statevar)
            self.logger.debug('pi_out:{}, V_out:{}'.format(pi_out.data, V_out.data))
            a = sampled_action(pi_out)
            self.past_action_log_props[self.t] = sampled_actions_log_probs(pi_out, a)
            self.past_entropy[self.t] = get_entropy(pi_out)
            self.past_Vs[self.t] = V_out
            self.t += 1
            return a[0]

        else:
            self.model.reset_state()
            return None



