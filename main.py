#!/usr/bin/env python3


# ___________________________________________ Libraries and Definitions ___________________________________________ #


# Learning libraries
import tensorflow as tf
import gym
from tensorboardX import SummaryWriter

# Math libraries
import numpy as np

# Time management libraries
import time

# Personal libraries
from HER import HER_Buffer, Experience
from HER_SAC_agent import HER_SAC_Agent


# ___________________________________________________ Parameters ___________________________________________________ #


# environment parameters
ENV_NAME = "FetchPush-v1"
LOG_DIR = "/home/gianfranco/Desktop/FetchLog"

# training parameters
TRAINING_EPOCHES = 1000
BATCH_SIZE = 50
MINIBATCH_SIZE = 256
RANDOM_EPISODES = 1000
HER_CAPACITY = 100000

# learning parameters
EPSILON_START = 0.5
EPSILON_FINAL = 0.01
EPSILON_DECAY_LAST_ITER = 1000000
LEARNING_RATE = 3e-4

# _____________________________________________________ Main _____________________________________________________ #


if __name__ == '__main__':

    # Environment initialization
    env = gym.make(ENV_NAME)

    # Agent initialization
    her_buff = HER_Buffer(HER_CAPACITY)
    agent = HER_SAC_Agent(env, her_buff)

    # Summary writer for live trends
    writer = SummaryWriter(log_dir=LOG_DIR, comment="LR"+ str(LEARNING_RATE))

    # Pre-training initialization
    iterations = 0
    loss_vect = []
    reward_vect = []
    epsilon = EPSILON_START

    # Training 
    for epoch in range(TRAINING_EPOCHES):
        print("\n\n************ TRAINING EPOCH ", epoch, "************\n")

        ## play batch
        for episode in range(BATCH_SIZE):
            print("Episode ", episode)
            epsilon = max(EPSILON_FINAL, EPSILON_START -
                          iterations / EPSILON_DECAY_LAST_ITER)
            experiences = agent.play_episode(criterion="SAC", epsilon=epsilon)
            reward_vect.append(exp.reward for exp in experiences)
            iterations += len(experiences)
            goal = experiences[-1].state['desired_goal']
            achieved_goal = experiences[-1].state['achieved_goal']
            reward = agent.env.compute_reward(achieved_goal, goal, None)
            agent.getBuffer().store_experience(experiences[-1], reward, goal)
            goal = achieved_goal
            for exp in experiences:
                reward = \
                    agent.env.compute_reward(exp.state['achieved_goal'], goal, None)
                agent.getBuffer().store_experience(exp, reward, goal)

        ## Minibatch sample and optimization 
        experiences = agent.getBuffer().sample(MINIBATCH_SIZE)
        v_loss, c1_loss, c2_loss, act_loss = \
            agent.optimization(experiences)

        ## print results
        m_reward_500 = np.mean(reward_vect[-500:])
        #m_loss_50 = np.mean(loss_vect[-50])
        print("Mean Reward [-500:] = ", m_reward_500)
        #print("Mean loss [-50:] = ", np.mean(loss_vect[-50]))
        writer.add_scalar("epsilon", epsilon, iterations)
        writer.add_scalar("mean_reward_500", m_reward_500, iterations)
        #writer.add_scalar("mean_loss_50", m_loss_50, iterations)
