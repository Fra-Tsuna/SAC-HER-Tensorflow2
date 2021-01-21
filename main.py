#!/usr/bin/env python3


# ___________________________________________ Libraries and Definitions ___________________________________________ #


# Learning libraries
import tensorflow as tf
import gym

# Math libraries
import numpy as np

# Time management libraries
import time

# Personal libraries
from HER import HER_Buffer, Experience


# ___________________________________________________ Parameters ___________________________________________________ #


# environment parameters
ENV_NAME = "FetchPush-v1"

# learning parameters
RANDOM_BATCH = 1000
HER_CAPACITY = 1000

# ______________________________________________ Classes and Functions ______________________________________________ #


class HER_SAC_Agent:

    def __init__(self, env, her_buffer):
        """
        Soft Actor Critic + Hindsight Experience Replay Agent
        
        Parameters
        ----------
        env: gym environment to solve
        her_buffer: Hindsight Experience Replay buffer
        """
        self.env = env
        self.her_buffer = her_buffer
        self.env.reset()

    def getBuffer(self):
        """
        Returns HER buffer of the agent
        """
        return self.her_buffer

    def random_play(self, batch_size):
        """
        Play a batch_size number of episode with random policy
        """
        for episode in range(batch_size):
            state = self.env.reset()
            print("Episode ", episode)
            step = 0
            while True:

                # Play
                step += 1
                self.env.render()
                action = self.env.action_space.sample()
                new_state, reward, done, _ = self.env.step(action)

                # Store in HER
                state_goal = np.concatenate([state['observation'], state['desired_goal']])
                new_state_goal = np.concatenate([new_state['observation'], state['desired_goal']])
                self.her_buffer.append(Experience(
                    state_goal, action, reward, new_state_goal, done))
                state = new_state
                print("\tStep: ", step, "Reward = ", reward)
                if reward > -1:
                    return True
                if done:
                    if episode == batch_size-1:
                        return False
                    break

# _____________________________________________________ Main _____________________________________________________ #


if __name__ == '__main__':

    # Environment initialization
    env = gym.make(ENV_NAME)
    obs_space = env.observation_space.shape
    n_actions = env.action_space.shape[0]

    # Agent initialization
    her_buff = HER_Buffer(HER_CAPACITY)
    agent = HER_SAC_Agent(env, her_buff)

    # Random playing (useful for testing)
    success = agent.random_play(RANDOM_BATCH)
    buffer = agent.getBuffer()
    if success:
        print("Goal achieved! Good job!")
    else:
        print("Bad play...")
    example_sample = buffer.sample()
    time.sleep(1)
    print("Example of sampling from the exp buffer: \n", example_sample)
