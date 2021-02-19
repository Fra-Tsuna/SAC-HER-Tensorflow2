#!/usr/bin/env python3

import random
import numpy as np
import tensorflow as tf
from collections import namedtuple, deque


""" 
Structure of a single Experience
N.B: the exp. stored in the HER buffer should have as state (new state) 
     the observation (new observation) concatenated with the goal
"""
Experience = namedtuple("Experience", field_names = \
    ['state', 'action', 'reward', 'new_state', 'done'])
    

class HER_Buffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        random.seed()

    def __len__(self):
        return len(self.buffer)

    def append(self, exp):
        self.buffer.append(exp)

    def store_experience(self, experience, reward, goal):
        """
        Store an experience in the HER buffer

        Parameters
        ----------
        experience: experience to store
        reward: the reward computed in the agent's env
        goal: the goal to concatenate to the states

        Return
        -------
        hindsight_exp: experience in the hindsight representation:
            (state||goal, action, reward, new_state||goal, done).
            N.B: || denotes concatenation
        """
        hindsight_exp = \
            self._hindsight_representation(experience, reward, goal)
        self.buffer.append(hindsight_exp)
        return hindsight_exp

    def sample(self, minibatch_size=1, ere_ck=None):
        """
        Sample items from the buffer

        Parameters
        ----------
        minibatch: number of items to sample from the buffer
        ere_ck: parameter ck of ERE algorithm which control sampling range

        Return
        -------
        items: hindsight experiences sampled from the buffer
        """
        items = []
        if ere_ck is None or ere_ck > len(self.buffer):
            sample_range = len(self.buffer)
        else:
            sample_range = int(ere_ck)
        locations = random.sample(
            range((len(self.buffer)-sample_range), len(self.buffer)), minibatch_size)
        for index in locations:
            items.append(self.buffer[index])
        return items

    def _hindsight_representation(self, experience, reward, goal):
        """
        Convert the passed experience to the HER canonical representation
        
        Parameters
        ----------
        experience: experience to convert
        goal: the goal obtained with any sampling strategy
        
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
