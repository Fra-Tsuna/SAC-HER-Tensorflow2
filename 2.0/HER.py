#!/usr/bin/env python3


import numpy as np
from collections import namedtuple, deque
import tensorflow as tf
import time


""" 
Structure of a single Experience
N.B: state can be anything (observation, dictionary, ecc.), but
     the exp. stored in the HER buffer should have as state the observation
     concatenated with the goal
"""
Experience = namedtuple("Experience", field_names = \
    ['state', 'action', 'reward', 'new_state', 'done'])


class HER_Buffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, exp):
        """
        Insert new element to the buffer (if full, old memory is dropped)

        Parameters
        ----------
        exp: experience to store = [state_goal, action, reward, newState_goal, done]
            N.B: _ denotes concatenation
        """
        self.buffer.append(exp)

    def store_experience(self, experience, reward, goal):
        """
        Store an experience in the HER buffer

        Parameters
        ----------
        experience: experience to store
        reward: the reward computed in the agent's env
        goal: the goal to concatenate to the states
        """
        hindsight_exp = self._hindsight_representation(experience, reward, goal)
        self.buffer.append(hindsight_exp)
        return hindsight_exp

    def sample(self, minibatch=1):
        """
        Sample items from the buffer

        Parameters
        ----------
        minibatch: number of items to sample from the buffer

        Returns
        -------
        items: items sampled from the buffer
        """
        locations = np.random.choice(len(self.buffer), minibatch, replace=False)            
        #if minibatch == 1:
        #    item = tf.convert_to_tensor(self.buffer[locations[0]])
        #    return item
        #else:
        states, actions, rewards, new_states, dones = [], [], [], [], []
        for index in locations:
            states.append(tf.convert_to_tensor(self.buffer[index].state))
            actions.append(tf.convert_to_tensor(self.buffer[index].action))
            rewards.append(tf.convert_to_tensor(self.buffer[index].reward))                     
            new_states.append(tf.convert_to_tensor(self.buffer[index].new_state))
            dones.append(tf.convert_to_tensor(1 - float(int(self.buffer[index].done))))

        return states, actions, rewards, new_states, dones


    def _hindsight_representation(self, experience, reward, goal):
        """
        Convert the experience in input into the her canonical representation
        
        Parameters
        ----------
        experience: experience to convert
        goal: the goal obtained with the opportune strategy
        
        Return
        ------
        experience converted
        """
        state = experience.state
        action = experience.action
        newState = experience.new_state
        done = experience.done
        state_goal = np.concatenate([state['observation'], goal])
        newState_goal = np.concatenate([newState['observation'], goal])
        return Experience(state_goal, action, reward, newState_goal, done)
