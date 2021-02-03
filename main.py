#!/usr/bin/env python3

# ___________________________________________________ Libraries ___________________________________________________ #


# Learning libraries
import tensorflow as tf
import gym
from tensorboardX import SummaryWriter

# Math libraries
import numpy as np
import random

# Personal libraries
from HER import HER_Buffer, Experience
from HER_SAC_agent import HER_SAC_Agent

# ___________________________________________________ Parameters ___________________________________________________ #


# environment parameters
ENV_NAME = "FetchReach-v1"
LOG_DIR = "/home/gianfranco/Desktop/FetchLog"

# training parameters
TRAINING_EPOCHES = 200
BATCH_SIZE = 50
OPTIMIZATION_STEPS = 40
POLICY_STEPS = 8
RANDOM_EPISODES = 1000
EVAL_EPISODES = 10

# HER parameters
HER_CAPACITY = 1000000
REPLAY_START_SIZE = 1000
MINIBATCH_SAMPLE_SIZE = 128
STRATEGY = "future"
FUTURE_K = 4

# learning parameters
EPSILON_START = 0.7

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
        print("\n\n************ TRAINING EPOCH ", epoch, "************\n")
        reward_vect = []

        ## play batch of cycles
        for cycle in range(BATCH_SIZE):
            print("Epoch ", epoch, "- Cycle ", cycle)
            hindsight_experiences = []

            ### play episodes
            for policy_step in range(POLICY_STEPS):
                #print("\t____Policy episode ", policy_step, "____")
                experiences = agent.play_episode(criterion="SAC", epsilon=epsilon)
                achieved_goal = experiences[-1].new_state['achieved_goal']
                goal = experiences[-1].state['desired_goal']
                true_reward = agent.env.compute_reward(achieved_goal, goal, None)
                iterations += len(experiences)
                for t in range(len(experiences)):
                    reward_vect.append(experiences[t].reward)
                    hindsight_exp = agent.getBuffer().store_experience(experiences[t], 
                                                                    true_reward, goal)
                    hindsight_experiences.append(hindsight_exp)
                    achieved_goal = experiences[t].new_state['achieved_goal']
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
                            desired_goal = experiences[future].new_state['achieved_goal']
                            achieved_goal = experiences[t].new_state['achieved_goal']
                            her_reward = \
                                agent.env.compute_reward(achieved_goal, desired_goal, None)
                            hindsight_exp = agent.getBuffer().store_experience(experiences[t],
                                                                    her_reward, desired_goal)
                            hindsight_experiences.append(hindsight_exp)    

                #### print results
                if iterations > 5000:
                    m_reward_5000 = np.mean(reward_vect[-5000:])
                    writer.add_scalar("mean_reward_5000", m_reward_5000, iterations)

            ### normalization
            agent.update_normalizer(batch=hindsight_experiences, hindsight=True)

            ### optimization
            if len(agent.getBuffer()) >= REPLAY_START_SIZE:
                epsilon = 0.2
                for opt_step in range(OPTIMIZATION_STEPS):
                    #print("__Optimization step ", opt_step, "__")
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
            success_rate = (len(experiences) - total_reward) / len(experiences)
            success_rates.append(success_rate)
        success_rate = sum(success_rates) / len(success_rates)
        print("Success_rate = ", success_rate)
        writer.add_scalar("success rate per epoch", success_rate, epoch)
