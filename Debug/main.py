#!/usr/bin/env python3

# ___________________________________________________ Libraries ___________________________________________________ #


# Learning
import gym
import tensorflow as tf
from tensorboardX import SummaryWriter

# Math 
import random
import numpy as np

# Custom libraries
from HER import HER_Buffer, Experience
from HER_SAC_agent import HER_SAC_Agent

# System
import os

# ___________________________________________________ Parameters ___________________________________________________ #


# Environment 
ENV_NAME = "FetchReach-v1"
LOG_DIR = "/home/gianfranco/Desktop/FetchLog"

# Training 
TRAINING_EPOCHES = 200
BATCH_SIZE = 50
OPTIMIZATION_STEPS = 40
POLICY_STEPS = 2
RANDOM_EPISODES = 1000
EVAL_EPISODES = 10

# HER 
HER_CAPACITY = 1000000
REPLAY_START_SIZE = 1000
MINIBATCH_SAMPLE_SIZE = 256
STRATEGY = "future"
FUTURE_K = 4

# Learning parameters
EPSILON_START = 0.5
EPSILON_NEXT = 0.1

# Debug 
DEBUG_EPISODE_EXP = False
DEBUG_STORE_EXP = False
DEBUG_MINIBATCH_SAMPLE = False

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
    epsilon = EPSILON_START

    # Training 
    for epoch in range(TRAINING_EPOCHES):
        #print("\n\n___________ TRAINING EPOCH ", epoch, "___________\n")
        reward_vect = []

        ## play batch of cycles
        for cycle in range(BATCH_SIZE):
            print("Epoch ", epoch, "- Cycle ", cycle)
            hindsight_experiences = []

            ### play episodes
            for policy_step in range(POLICY_STEPS):
                experiences = agent.play_episode(criterion="SAC", epsilon=epsilon)
                goal = experiences[0].state['desired_goal']
                if DEBUG_EPISODE_EXP:
                    print("\n\n++++++++++++++++ DEBUG - EPISODE EXPERIENCES [MAIN.POLICY_STEPS] +++++++++++++++++\n")
                    print("----------------------------len experiences----------------------------")
                    print(len(experiences))
                    print("----------------------------last experience----------------------------")
                    print(experiences[-1])
                    print("----------------------------desired goal----------------------------")
                    print(goal)
                    a = input("\n\nPress Enter to continue...")
                iterations += len(experiences)
                for t in range(len(experiences)):
                    reward_vect.append(experiences[t].reward)
                    achieved_goal = experiences[t].new_state['achieved_goal']
                    hindsight_exp = agent.getBuffer().store_experience(experiences[t], 
                                                                    experiences[t].reward, goal)
                    hindsight_experiences.append(hindsight_exp)
                    if DEBUG_STORE_EXP:
                        print("\n\n++++++++++++++++ DEBUG - STORE EXPERIENCES [MAIN.POLICY_STEPS] +++++++++++++++++\n")
                        print("----------------------------actual experience----------------------------")
                        print(experiences[t])
                        print("----------------------------hindsight experience true----------------------------")
                        print(hindsight_exp)
                    if STRATEGY == "final":
                        desired_goal = experiences[-1].new_state['achieved_goal']
                        her_reward = \
                            agent.env.compute_reward(achieved_goal, desired_goal, None)
                        hindsight_exp = agent.getBuffer().store_experience(experiences[t], 
                                                                her_reward, desired_goal)
                        hindsight_experiences.append(hindsight_exp)
                    elif STRATEGY == "future":
                        for f_t in range(FUTURE_K):
                            future = random.randint(t, (len(experiences)-1))
                            if DEBUG_STORE_EXP:
                                print("----------------------------future index----------------------------")
                                print(future)
                            desired_goal = experiences[future].new_state['achieved_goal']
                            her_reward = \
                                agent.env.compute_reward(achieved_goal, desired_goal, None)
                            hindsight_exp = agent.getBuffer().store_experience(experiences[t],
                                                                    her_reward, desired_goal)
                            hindsight_experiences.append(hindsight_exp)
                            if DEBUG_STORE_EXP:
                                print("----------------------------hindsight experience future----------------------------")
                                print(hindsight_exp)
                                a = input("\n\nPress Enter to continue...")
                    else:
                        raise TypeError("Wrong strategy for goal sampling. \
                                        [available 'final' or 'future']")   
                        
                #### print results
                mean_reward = np.mean(reward_vect)
                writer.add_scalar("mean_reward", mean_reward, iterations)

            ### normalization
            agent.update_normalizer(batch=hindsight_experiences, hindsight=True)

            ### optimization
            if len(agent.getBuffer()) >= REPLAY_START_SIZE:
                epsilon = EPSILON_NEXT
                for opt_step in range(OPTIMIZATION_STEPS):
                    minibatch = agent.getBuffer().sample(MINIBATCH_SAMPLE_SIZE)
                    v_loss, c1_loss, c2_loss, act_loss = \
                        agent.optimization(minibatch)
                    agent.soft_update()

        ## evaluation
        print("\n\nEVALUATION\n\n")
        success_rates = []
        for _ in range(EVAL_EPISODES):
            experiences = agent.play_episode(criterion="SAC", epsilon=0)
            total_reward = sum([exp.reward for exp in experiences])
            success_rate = (len(experiences) + total_reward) / len(experiences)
            success_rates.append(success_rate)
        success_rate = sum(success_rates) / len(success_rates)
        print("Success_rate = ", success_rate)
        writer.add_scalar("mean success rate per epoch", success_rate, epoch)
