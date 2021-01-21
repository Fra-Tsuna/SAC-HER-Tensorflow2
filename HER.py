#!/usr/bin/env python3

import numpy as np
from collections import namedtuple, deque


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
        exp: experience to store = [state||goal, action, reward, new_state||goal, done]
        """
        self.buffer.append(exp)

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
        if minibatch == 1:
            items = self.buffer[locations[0]]
        else:
            items = []
            for index in locations:
                items.append(self.buffer[index])
        return items
