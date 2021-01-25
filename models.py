#!/usr/bin/env python3


import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import \
    Model, Sequential, Input, layers


# Hyperparameters
ACTOR_DENSE_1 = 256
ACTOR_DENSE_2 = 256
CRITIC_DENSE_1 = 256
CRITIC_DENSE_2 = 256
VALUE_DENSE_1 = 256
VALUE_DENSE_2 = 256

NOISE = 1e-6


class ActorNetwork(Model):

    def __init__(self, input_dim, action_dim):
        super(ActorNetwork, self).__init__()

        self.net = Sequential()
        self.net.add(layers.Input(shape=input_dim))
        self.net.add(layers.Dense((ACTOR_DENSE_1)))
        self.net.add(layers.ReLU())
        self.net.add(layers.Dense(ACTOR_DENSE_2))
        self.net.add(layers.ReLU())

        self.mean = self.net.add(layers.Dense(action_dim))
        self.std_dev = self.net.add(layers.Dense(action_dim))

    def forward(self, state, noisy=True):

        # policy parameters 
        mi = self.mean(state)
        #sigma = np.clip(self.std_dev(state), NOISE, 1)         ?
        sigma = tf.clip_by_value(self.std_dev(state), NOISE, 1)
        policy = tfp.distributions.Normal(mi, sigma)
        noise = tfp.distributions.Normal(0,1).sample()

        # action selection
        if noisy:
            action_sample = noise + policy.sample()
        else:
            action_sample = policy.sample()
        action = tf.tanh(actions_tensor)
        log_probs = policy.log_prob(actions_tensor)

        # enforcing action bounds
        log_probs -= \
            tf.reduce_sum(tf.math.log(1 - action**2 + NOISE), axis=1, keepdims=True)

        return action, log_probs

class CriticNetwork(Model):

    def __init__(self, input_dim):
        super(CriticNetwork, self).__init__()

        self.net = Sequential()
        self.net.add(Input(shape=input_dim))
        self.net.add(layers.Dense(CRITIC_DENSE_1))
        self.net.add(layers.ReLU())
        self.net.add(layers.Dense(CRITIC_DENSE_2))
        self.net.add(layers.ReLU())
        self.net.add(layers.Dense(1))

    def forward(self, state, action):
        state_action = tf.concat([state, action], axis=1)
        q_value = self.net(state_action)
        return q_value


class ValueNetwork(Model):

    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()

        self.net = Sequential()
        self.net.add(Input(shape=input_dim))
        self.net.add(layers.Dense(VALUE_DENSE_1))
        self.net.add(layers.ReLU())
        self.net.add(layers.Dense(VALUE_DENSE_2))
        self.net.add(layers.ReLU())
        self.net.add(layers.Dense(1))
    
    def forward(self, state):
        value = self.net(state)
        return value
