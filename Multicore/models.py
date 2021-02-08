#!/usr/bin/env python3

import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import clone_model
from tensorflow.keras import Model, Sequential, layers


# Hyperparameters
ACTOR_DENSE_1 = 256
ACTOR_DENSE_2 = 256
CRITIC_DENSE_1 = 256
CRITIC_DENSE_2 = 256
VALUE_DENSE_1 = 256
VALUE_DENSE_2 = 256

NOISE = 1e-8


class ActorNetwork(Model):

    def __init__(self, input_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.input_layer = layers.InputLayer(input_shape=input_dim)
        self.layer_1 = layers.Dense(ACTOR_DENSE_1, activation=layers.ReLU())
        self.layer_2 = layers.Dense(ACTOR_DENSE_2, activation=layers.ReLU())
        self.mean = layers.Dense(action_dim)
        self.log_std_dev = layers.Dense(action_dim)

    def call(self, state, noisy=True):
        x = self.layer_2(self.layer_1(self.input_layer(state)))
        mean = self.mean(x)
        log_std = self.log_std_dev(x)
        log_std_clipped = tf.clip_by_value(log_std, NOISE, 1)
        policy = tfp.distributions.Normal(mean, tf.exp(log_std_clipped))
        noise = tfp.distributions.Normal(0,1).sample()
        if noisy:
            action_sample = mean + noise*tf.exp(log_std_clipped)
        else:
            action_sample = policy.sample()
        squashed_actions = tf.tanh(action_sample)
        logprob = (policy.log_prob(action_sample) - 
                   tf.math.log(1.0 - tf.pow(squashed_actions, 2) + NOISE))
        logprob = tf.reduce_sum(logprob, axis=-1, keepdims=True)
        return squashed_actions, logprob


class CriticNetwork(Model):

    def __init__(self, input_dim):
        super(CriticNetwork, self).__init__()

        self.net = Sequential()
        self.net.add(layers.InputLayer(input_shape=input_dim))
        self.net.add(layers.Dense(CRITIC_DENSE_1))
        self.net.add(layers.ReLU())
        self.net.add(layers.Dense(CRITIC_DENSE_2))
        self.net.add(layers.ReLU())
        self.net.add(layers.Dense(1))

    def call(self, state, action):
        state_action = tf.concat([state, action], axis=1)
        q_value = self.net(state_action)
        return q_value


class ValueNetwork(Model):

    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()

        self.net = Sequential()
        self.net.add(layers.InputLayer(input_shape=input_dim))
        self.net.add(layers.Dense(VALUE_DENSE_1))
        self.net.add(layers.ReLU())
        self.net.add(layers.Dense(VALUE_DENSE_2))
        self.net.add(layers.ReLU())
        self.net.add(layers.Dense(1))
    
    def call(self, state):
        value = self.net(state)
        return value
