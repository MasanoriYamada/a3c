#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import threading
import signal
import datetime
import os
import numpy as np
import gym

from a3c.agent_a3c import Agent_a3c

from tensorboard import SummaryWriter
from a3c.constants import MAX_EPISODE_LEN
from a3c.constants import N_THREADS
from a3c.constants import N_TEST_RUN
from a3c.constants import EVAL_INTERVAL

def evaluate(env, agent):
    logger = logging.getLogger(__name__)
    logger.info('evaluate')

    r = 0
    local_r_sum = 0.
    local_r_sum_lst = []
    episode = 0
    done = False
    obs = env.reset()

    while True:
        if episode == N_TEST_RUN:
            logger.info('test r_sum: {}, test num: {}'.format(np.mean(np.array(local_r_sum_lst)), N_TEST_RUN))
            break

        if done:
            local_r_sum_lst.append(local_r_sum)
            local_r_sum = 0
            obs = env.reset()
            episode += 1
        else:
            # Todo impliment agent.act which is no learning mode
            a = agent.act_and_train(obs, r, is_state_terminal=done)
            obs, r, done, info = env.step(a)
            local_r_sum += r

def async_train(env_name, shared_model, opt, phi):
    logger = logging.getLogger(__name__)
    board_path = 'results/' + datetime.datetime.now().strftime('%B%d  %H:%M:%S')
    training_threads = []
    for i in range(N_THREADS):
        training_threads.append(threading.Thread(target=train_loop, args=(i, env_name, shared_model, opt, phi, board_path)))

    for i, th in enumerate(training_threads):
        logger.info('start thread_id : {}'.format(i))
        th.start()

    print('Press Ctrl+C to stop')
    signal.pause()
    print('Now saving data. Please wait')

    for th in training_threads:
        th.join()

    # model save
    output_model_path = 'results/' + datetime.now().strftime('%B%d  %H:%M:%S')
    if not os.path.exists(output_model_path):
        os.mkdir(output_model_path)
    #agent.save_model(output_model_path)
    logger.info('model saved')

def train_loop(thread_id, env_name, shared_model, opt, phi, board_path):
    logger = logging.getLogger(__name__)
    agent = Agent_a3c(shared_model=shared_model, optimizer=opt, phi=phi)
    agent.generagte_local_model(thread_id)

    done = False
    episode = 0
    r = 0
    step = 0
    global_step = 0
    local_r_sum = 0
    env = gym.make(env_name)
    obs = env.reset()

    # set writer

    writer = SummaryWriter(board_path)

    while True:
        if done or step == MAX_EPISODE_LEN:
            obs = env.reset()
            global_step += step
            if thread_id == 0:
                logger.info('episode: {}, r_sum: {}, total_step:{}, step len in episode: {}'.format(episode, local_r_sum, step, step - step_last))
                if episode % EVAL_INTERVAL == 0:
                    evaluate(env, agent)
            writer.add_scalar('reward_sum_{}'.format(thread_id), local_r_sum, episode)
            writer.add_scalar('V_{}'.format(thread_id), agent.shared_V_out.data, episode)
            writer.add_scalar('A_{}'.format(thread_id), agent.A.data, episode)
            writer.add_scalar('loss_v_{}'.format(thread_id), agent.V_loss.data, episode)
            writer.add_scalar('loss_pi_{}'.format(thread_id), agent.pi_loss.data, episode)
            writer.add_scalar('loss_entropy_{}'.format(thread_id), agent.entropy_loss.data, episode)
            writer.add_all_parameter_histograms([agent.pi_loss], episode)
            writer.add_all_parameter_histograms([agent.V_loss], episode)
            writer.add_all_parameter_histograms([agent.entropy_loss], episode)

            r = 0
            step_last = step
            local_r_sum = 0
            done = False
            episode += 1

        else:
            a = agent.act_and_train(obs, r, is_state_terminal=done)
            obs, r, done, info = env.step(a)
            r = 0.01 * r  # reduce
            if thread_id == 0:
                logger.debug('step: {}, r: {}, a: {}, s: {}, done: {}, info: {}'.format(step, r, a, obs, done, info))
            local_r_sum += r
            step += 1
