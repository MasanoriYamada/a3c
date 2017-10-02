#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import threading
import signal
import datetime
import os
import numpy as np

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

def async_train(env, agent):
    logger = logging.getLogger(__name__)

    training_threads = []
    for i in range(N_THREADS):
        training_threads.append(threading.Thread(target=train_loop, args=(i, env, agent)))

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
    agent.save_model(output_model_path)
    logger.info('model saved')

def train_loop(thread_id, env, agent):
    logger = logging.getLogger(__name__)

    done = False
    episode = 0
    r = 0
    step = 0
    global_step = 0
    local_r_sum = 0
    obs = env.reset()

    # set writer
    if thread_id == 0:
        writer = SummaryWriter('results/' + datetime.datetime.now().strftime('%B%d  %H:%M:%S'))

    while True:
        if done or step == MAX_EPISODE_LEN:
            obs = env.reset()
            global_step += step
            if thread_id == 0:
                logger.info('episode: {}, r_sum: {}, total step in episode: {}'.format(episode, local_r_sum, step))
                writer.add_scalar('reward_sum', local_r_sum, episode)
                if episode % EVAL_INTERVAL == 0:
                    evaluate(env, agent)
            r = 0
            step = 0
            local_r_sum = 0
            done = False
            episode += 1

        else:
            a = agent.act_and_train(obs, r, is_state_terminal=done)
            obs, r, done, info = env.step(a)
            if thread_id == 0:
                logger.debug('step: {}, r: {}, a: {}, s: {}, done: {}, info: {}'.format(step, r, a, obs, done, info))
            local_r_sum += r
            step += 1
