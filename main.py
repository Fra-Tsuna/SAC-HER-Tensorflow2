#!/usr/bin/env python3


# ___________________________________________________ Libraries ___________________________________________________ #


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
TRAINING_EPOCHES = 200
BATCH_SIZE = 50
OPTIMIZATION_STEPS = 40
POLICY_STEPS = 16
MINIBATCH_SAMPLE_SIZE = 128
RANDOM_EPISODES = 1000
HER_CAPACITY = 1000000

# learning parameters
EPSILON_START = 0.3
EPSILON_FINAL = 0.01
EPSILON_DECAY_LAST_ITER = 30000

# _____________________________________________________ Main _____________________________________________________ #


if __name__ == '__main__':

    # Environment initialization
    env = gym.make(ENV_NAME)

    # Agent initialization
    her_buff = HER_Buffer(HER_CAPACITY)
    agent = HER_SAC_Agent(env, her_buff)

    # Summary writer for live trends
    writer = SummaryWriter(log_dir=LOG_DIR, comment="NoComment")

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

            ### play episode
            for policy_step in range(POLICY_STEPS):
                print("Epoch ", epoch, "- Episode ", episode)
                print("\t____Policy step ", policy_step, "____")
                epsilon = max(EPSILON_FINAL, EPSILON_START -
                            iterations / EPSILON_DECAY_LAST_ITER)
                experiences = agent.play_episode(criterion="SAC", epsilon=epsilon)
                iterations += len(experiences)
                goal = experiences[-1].state['desired_goal']
                achieved_goal = experiences[-1].state['achieved_goal']
                true_reward = agent.env.compute_reward(achieved_goal, goal, None)
                for exp in experiences:
                    reward_vect.append(exp.reward)
                    agent.getBuffer().store_experience(exp, true_reward, goal)
                    her_reward = agent.env.compute_reward(exp.state['achieved_goal'], 
                                                          achieved_goal, None)
                    agent.getBuffer().store_experience(exp, her_reward, achieved_goal)
                
                #### print results
                if iterations > 1000:
                    m_reward_1000 = np.mean(reward_vect[-1000:])
                    writer.add_scalar("epsilon", epsilon, iterations)
                    writer.add_scalar("mean_reward_1000", m_reward_1000, iterations)

            ### optimization
            for opt_step in range(OPTIMIZATION_STEPS):
                print("__Optimization step ", opt_step, "__")
                experiences = agent.getBuffer().sample(MINIBATCH_SAMPLE_SIZE)
                v_loss, c1_loss, c2_loss, act_loss = \
                    agent.optimization(experiences)

        ## success rate
        total_success = 0
        for rew in reward_vect:
            total_success += 1 if rew > -1 else 0
        success_rate = total_success/len(reward_vect)
        writer.add_scalar("success rate per epoch", success_rate, epoch)
        reward_vect = []
