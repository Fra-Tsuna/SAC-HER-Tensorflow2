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
        self.env = env
        self.her_buffer = her_buffer
        self.env.reset()

    def getBuffer(self):
        return self.her_buffer

    def play_episode(self, criterion="random", store=False):
        """
        Play an episode choosing actions according to the selected criterion
        
        Parameters
        ----------
        criterion: strategy to choose actions ('random' or 'SAC')
        store: if True, all the experiences are stored in the HER buffer

        Returns
        -------
        last_state: state in which the agent stands after last action
        final_reward: last reward obtained by the agent
        """
        state = self.env.reset()
        last_state = state
        final_reward = None
        done = False
        while not done:

            # Play
            self.env.render()
            if criterion == "random":
                action = self.env.action_space.sample()
            elif criterion == "SAC":
                #action = ACTION SELECTED WITH SAC
            else:
                print("Wrong criterion for choosing the action")
            new_state, reward, done, _ = self.env.step(action)

            # Store in HER
            if store:
                state_goal = np.concatenate([state['observation'], state['desired_goal']])
                new_state_goal = np.concatenate([new_state['observation'], 
                                                state['desired_goal']])
                self.her_buffer.append(
                    Experience(state_goal, action, reward, new_state_goal, done))
            state = new_state
            if done:
                last_state = new_state
                final_reward = reward
            print("\tStep: ", step, "Reward = ", reward)
        return last_state, final_reward

    #def train(self, batch_size):
    #    """
    #    Train the agent with a batch_size number of episodes
    #    """
    #    for episode in batch_size:
            

    def random_play(self, batch_size):
        """
        Play a batch_size number of episode with random policy
        Returns True if the agent reach the goal
        """
        for episode in range(batch_size):
            state = self.env.reset()
            print("Random episode ", episode)
            actual_state, actual_reward = self.play_episode()
            if actual_reward > -1:
                return True
        return False



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
