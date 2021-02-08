#!/usr/bin/env python3

import time
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

# debug parameters
DEBUG_EXP = False
DEBUG_BUFFER = False
DEBUG_SAMPLE = False


class HER_Buffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

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
        if DEBUG_BUFFER:
            print("\n\n\t++++++++++++++++ DEBUG - BUFFER [HER.STORE_EXP] +++++++++++++++++\n")
            print("\t----------------------------len----------------------------")
            print("\t", len(self.buffer))
            a = input("\tPress Enter to continue...")
        return hindsight_exp

    def sample(self, minibatch_size=1):
        """
        Sample items from the buffer

        Parameters
        ----------
        minibatch: number of items to sample from the buffer

        Return
        -------
        items: hindsight experiences sampled from the buffer
        """
        locations = np.random.choice(len(self.buffer), 
                        minibatch_size, replace=False)
        items = []
        if minibatch_size == 1:
            items = self.buffer[locations[0]]
        else:
            for index in locations:
                items.append(self.buffer[index])
        if DEBUG_SAMPLE:
            print("\n\n\t++++++++++++++++ DEBUG - SAMPLE [HER.SAMPLE] +++++++++++++++++\n")
            print("\t----------------------------indexes----------------------------")
            print("\t", locations)
            a = input("\n\n\tPress Enter to continue...")
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
        if DEBUG_EXP:
            print("\n\n\t++++++++++++++++ DEBUG - STORE EXPERIENCES [HER.HINDSIGHT_REPRESENTATION] +++++++++++++++++\n")
            print("\t----------------------------state||goal----------------------------")
            print("\t", state_goal)
            print("\t----------------------------action----------------------------")
            print("\t", action)
            print("\t----------------------------her reward----------------------------")
            print("\t", reward)
            print("\t----------------------------new_state||goal----------------------------")
            print("\t", newState_goal)
            print("\t----------------------------done----------------------------")
            print("\t", done)
            a = input("\n\n\tPress Enter to continue...")
        return Experience(state_goal, action, reward, newState_goal, done)
