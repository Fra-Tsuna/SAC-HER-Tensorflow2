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

# ___________________________________________________ Parameters ___________________________________________________ #


# Environment 
ENV_NAME = "FetchPush-v1"
GIANFRANCO_LOG_DIR = "/home/gianfranco/Desktop/FetchLog"
FRANCESCO_LOG_DIR = "/home/francesco/Scrivania/Test"
LOG_DIR = GIANFRANCO_LOG_DIR
EPISODE_LEN = 50
ENV_WRAPPED = True

# Training 
TRAINING_EPOCHES = 200
BATCH_SIZE = 50
CYCLE_EPISODES = 1
TRAINING_START_STEPS = 1000
OPTIMIZATION_STEPS = 50
EVAL_EPISODES = 10

# HER 
HER_CAPACITY = 1000000
MINIBATCH_SAMPLE_SIZE = 256
STRATEGY = "future"
FUTURE_K = 4

# ERE
CMIN = 5000
ETA = 0.922

# Learning parameters
EPSILON_START = 1.
EPSILON_NEXT = 0.
TEMPERATURE = "auto"

# ____________________________________________________ Classes ____________________________________________________ #


class DoneOnSuccessWrapper(gym.Wrapper):
    """
    Reset on success and offsets the reward.
    Useful for GoalEnv.
    """
    def __init__(self, env, reward_offset=1.0):
        super(DoneOnSuccessWrapper, self).__init__(env)
        self.reward_offset = reward_offset

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        done = done or info.get('is_success', False)
        reward += self.reward_offset
        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = self.env.compute_reward(achieved_goal, desired_goal, info)
        return reward + self.reward_offset

# _____________________________________________________ Main _____________________________________________________ #


if __name__ == '__main__':

    # Environment initialization
    env = gym.make(ENV_NAME)
    if ENV_WRAPPED:
        env = DoneOnSuccessWrapper(env)

    # Agent initialization
    her_buff = HER_Buffer(HER_CAPACITY)
    agent = HER_SAC_Agent(env, her_buff, temperature=TEMPERATURE)

    # Summary writer for live trends
    writer = SummaryWriter(log_dir=LOG_DIR, comment="NoComment")

    # Pre-training initialization
    iterations = 0
    loss_vect = []
    reward_vect = []
    epsilon = EPSILON_START

    # Training 
    for epoch in range(TRAINING_EPOCHES):
        #print("\n\n___________ TRAINING EPOCH ", epoch, "___________\n")
        box_displ = 0

        ## play batch of cycles
        for cycle in range(BATCH_SIZE):
            print("Epoch ", epoch, "- Cycle ", cycle)
            hindsight_experiences = []
            played_experiences = 0

            ### play episodes
            for episode in range(CYCLE_EPISODES):
                experiences = agent.play_episode(criterion="SAC", epsilon=epsilon)
                goal = experiences[0].state['desired_goal']
                iterations += len(experiences)
                played_experiences += len(experiences)
                if (agent.env.compute_reward(experiences[0].new_state['achieved_goal'], 
                                             experiences[-1].new_state['achieved_goal'], 
                                             None)) == 0:
                    box_displ += 1
                for t in range(len(experiences)):
                    reward_vect.append(experiences[t].reward)
                    achieved_goal = experiences[t].new_state['achieved_goal']
                    hindsight_exp = agent.getBuffer().store_experience(experiences[t], 
                                                                    experiences[t].reward, goal)
                    hindsight_experiences.append(hindsight_exp)
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
                            her_reward = \
                                agent.env.compute_reward(achieved_goal, desired_goal, None)
                            hindsight_exp = agent.getBuffer().store_experience(experiences[t],
                                                                    her_reward, desired_goal)
                            hindsight_experiences.append(hindsight_exp)
                    else:
                        raise TypeError("Wrong strategy for goal sampling." +
                                        " [available 'final', 'future']")   

            ### print results
            if iterations > 2500:
                mean_reward = np.mean(reward_vect[-2500:])
                writer.add_scalar("mean_reward", mean_reward, iterations)

            ### normalization
            agent.update_normalizer(batch=hindsight_experiences, hindsight=True)

            ### optimization + ERE 
            if iterations >= TRAINING_START_STEPS:
                k = 0
                epsilon = EPSILON_NEXT
                opt_steps = min(played_experiences, OPTIMIZATION_STEPS)
                for step in range(opt_steps):
                    ck = max(HER_CAPACITY*pow(ETA, k*(EPISODE_LEN/opt_steps)), CMIN)
                    v_loss, c1_loss, c2_loss, act_loss, temp_loss = \
                        agent.optimization()
                    agent.soft_update()
                    k += 1
            if TEMPERATURE == "auto":
                print("\n\tTemperature: ", agent.getTemperature())                  
                writer.add_scalar("temperature", agent.getTemperature(), iterations)

        ## evaluation
        if ENV_NAME == "FetchPush-v1":
            print("\tBox displacements = ", box_displ)
        print("\n\nEVALUATION\n\n")
        success_rates = []
        for _ in range(EVAL_EPISODES):
            experiences = agent.play_episode(criterion="SAC", epsilon=0)
            total_reward = sum([exp.reward for exp in experiences])
            if ENV_WRAPPED:
                success_rate = total_reward / len(experiences)
            else:
                success_rate = (len(experiences) + total_reward) / len(experiences)
            success_rates.append(success_rate)
        success_rate = sum(success_rates) / len(success_rates)
        print("Success_rate = ", success_rate)
        if ENV_NAME == "FetchPush-v1":
            writer.add_scalar("box displacements per epoch", box_displ, epoch)
        writer.add_scalar("mean success rate per epoch", success_rate, epoch)
