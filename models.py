#!/usr/bin/env python3


import numpy as np
import tensorflow as tf
from tf.keras import \
    Model, Sequential, Input, layers
from tf.keras.optimizers import Adam


# Hyperparameters
CRITIC_DENSE_1 = 128
CRITIC_DENSE_2 = 128


class CriticNetwork(Model):

    def __init__(self, beta, input_dim):
        super.__init__()

        self.net = Sequential()
        self.net.add(Input(shape=input_dim))
        self.net.add(layers.Dense(CRITIC_DENSE_1))
        self.net.add(layers.ReLU())
        self.net.add(layers.Dense(CRITIC_DENSE_2))
        self.net.add(layers.ReLU())
        self.net.add(layers.Dense(1))

        def forward(self, state, action):
            state_action = tf.concat([state, action], axis=1)
            out = self.net(state_action)
            return out
        
class ValueNetwork():
    def __init__(self, beta, input_dims, l1_dims=256, l2_dims=256):
        self.input_dims=input_dims
        self.l1_dims=l1_dims
        self.l2_dims=l2_dims

        self.model=Sequential()
        self.model.add(Input(*self.input_dims,))
        self.model.add(Dense(self.l1_dims))
        self.model.add(ReLU)
        self.model.add(Dense(self.l2_dims))
        self.model.add(ReLU)
        self.model.add(Dense(1))
        self.optimizer=Adam(learning_rate=beta)
        self.model.compile(optimizer=optimizer)
    
    def forward(self, state):
        return self.model.predict(state)
