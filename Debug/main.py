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
POLICY_STEPS = 2
RANDOM_EPISODES = 1000
EVAL_EPISODES = 10

# HER parameters
HER_CAPACITY = 1000000
REPLAY_START_SIZE = 1000
MINIBATCH_SAMPLE_SIZE = 256
STRATEGY = "future"
FUTURE_K = 4

# learning parameters
EPSILON_START = 0.5

# debug parameters
DEBUG_EPISODE_EXP = False
DEBUG_STORE_EXP = False
DEBUG_MINIBATCH_SAMPLE = False
DEBUG_HINDSIGHT_EXP = False

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
                goal = experiences[0].state['desired_goal']
                if DEBUG_EPISODE_EXP:
                    print("\n\n++++++++++++++++ DEBUG - EPISODE EXPERIENCES [MAIN.POLICY_STEPS] +++++++++++++++++\n")
                    print("----------------------------len experiences----------------------------")
                    print(len(experiences))
                    print("----------------------------last experience----------------------------")
                    print(experiences[-1])
                    print("----------------------------achieved goal----------------------------")
                    print(achieved_goal)
                    print("----------------------------desired goal----------------------------")
                    print(goal)
                    print("----------------------------last reward----------------------------")
                    print(true_reward)
                    a = input("\n\nPress Enter to continue...")
                iterations += len(experiences)
                if DEBUG_STORE_EXP:
                    print("\n\n++++++++++++++++ DEBUG - STORE EXPERIENCES [MAIN.POLICY_STEPS] +++++++++++++++++\n")
                for t in range(len(experiences)):
                    reward_vect.append(experiences[t].reward)
                    achieved_goal = experiences[t].new_state['achieved_goal']
                    #true_reward = agent.env.compute_reward(achieved_goal, goal, None)
                    hindsight_exp = agent.getBuffer().store_experience(experiences[t], 
                                                                    experiences[t].reward, goal)
                    hindsight_experiences.append(hindsight_exp)
                    if DEBUG_STORE_EXP:
                        print("----------------------------actual t----------------------------")
                        print(t)
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
                    if DEBUG_STORE_EXP:
                        a = input("\n\nPress Enter to continue...")     

                #### print results
                mean_reward = np.mean(reward_vect)
                writer.add_scalar("mean_reward", mean_reward, iterations)

            ### normalization
            if DEBUG_HINDSIGHT_EXP:
                    print("\n\n++++++++++++++++ DEBUG - HINDSIGHT EXPERIENCES [MAIN.CYCLE] +++++++++++++++++\n")
                    print("----------------------------len exp----------------------------")
                    print(len(hindsight_experiences))
                    print("----------------------------element 0----------------------------")
                    print(hindsight_experiences[0])
                    print("----------------------------element -1----------------------------")
                    print(hindsight_experiences[-1])
                    print("----------------------------random element----------------------------")
                    print(hindsight_experiences[random.randint(0, len(hindsight_experiences)-1)])
                    print("----------------------------positive rewards----------------------------")
                    positive_rewards = [exp.reward for exp in hindsight_experiences if exp.reward > -1]
                    print(len(positive_rewards))
                    a = input("\n\nPress Enter to continue...")
            agent.update_normalizer(batch=hindsight_experiences, hindsight=True)

            ### optimization
            if len(agent.getBuffer()) >= REPLAY_START_SIZE:
                epsilon = 0.2
                for opt_step in range(OPTIMIZATION_STEPS):
                    #print("__Optimization step ", opt_step, "__")
                    minibatch = agent.getBuffer().sample(MINIBATCH_SAMPLE_SIZE)
                    if DEBUG_MINIBATCH_SAMPLE:
                        print("\n\n++++++++++++++++ DEBUG - MINIBATCH SAMPLE [MAIN.OPTIMIZATION_STEPS] +++++++++++++++++\n")
                        print("----------------------------len minibatch----------------------------")
                        print(len(minibatch))
                        print("----------------------------element 0----------------------------")
                        print(minibatch[0])
                        print("----------------------------element -1----------------------------")
                        print(minibatch[-1])
                        print("----------------------------random element----------------------------")
                        print(minibatch[random.randint(0, len(minibatch)-1)])
                        a = input("\n\nPress Enter to continue...")
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
        writer.add_scalar("success rate per epoch", success_rate, epoch)
