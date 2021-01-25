#!/usr/bin/env python3


from HER import HER_Buffer, Experience
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
from models import ActorNetwork, CriticNetwork, ValueNetwork
from tensorboardX import SummaryWriter


# Learning parameters
LEARNING_RATE = 3e-4
GAMMA = 0.99
TAU = 0.005


class HER_SAC_Agent:

    def __init__(self, env, her_buffer, optimizer='Adam'):
        self.env = env
        self.her_buffer = her_buffer
        self.env.reset()
        self.actor = \
            ActorNetwork(env.observation_space.shape, env.action_space.shape[0])
        self.critic_1 = \
            CriticNetwork(env.observation_space.shape)
        self.critic_2 = \
            CriticNetwork(env.observation_space.shape)
        self.value = \
            ValueNetwork(env.observation_space.shape)
        self.target_value = \
            ValueNetwork(env.observation_space.shape)
        if optimizer == 'Adam':
            self.optimizer = Adam(learning_rate=LEARNING_RATE)
        else:
            self.optimizer = None
            print("Error: Wrong optimizer for the agent")

    def getBuffer(self):
        return self.her_buffer

    def play_episode(self, criterion="random", epsilon=0):
        """
        Play an episode choosing actions according to the selected criterion
        
        Parameters
        ----------
        criterion: strategy to choose actions ('random' or 'SAC')
        epsilon: random factor for epsilon-greedy strategy
        Returns
        -------
        experiences: all experiences taken by the agent in the episode 
        """
        state = self.env.reset()
        experiences = []
        done = False
        step = 0
        while not done:
            step += 1
            self.env.render()
            if criterion == "random":
                action = self.env.action_space.sample()
            elif criterion == "SAC":
                if np.random.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    action, _ = self.actor.forward(state)
                    #action = action[0]                             ?
            else:
                print("ERROR: Wrong criterion for choosing the action")
            new_state, reward, done, _ = self.env.step(action)
            experiences.append(Experience(state, action, reward, new_state, done))
            state = new_state
            print("\tStep: ", step, "Reward = ", reward)
        return experiences

    def random_play(self, batch_size):
        """
        Play a batch_size number of episode with random policy
        Returns True if the agent reach the goal
        """
        for episode in range(batch_size):
            state = self.env.reset()
            print("Random episode ", episode)
            experiences = self.play_episode()
            if experiences[-1].reward > -1:
                return True
        return False
