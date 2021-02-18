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

NOISE = 1e-6
LOG_STD_MIN = -20
LOG_STD_MAX = 2

# debug parameters
DEBUG_ACTOR_NET = False
DEBUG_ACTOR_SAMPLE = False
DEBUG_VALUE = False
DEBUG_TARGET = False


class ActorNetwork(Model):

    def __init__(self, input_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.input_layer = layers.InputLayer(input_shape=input_dim)
        self.layer_1 = layers.Dense(ACTOR_DENSE_1, activation=layers.ReLU())
        self.layer_2 = layers.Dense(ACTOR_DENSE_2, activation=layers.ReLU())
        self.mean = layers.Dense(action_dim)
        self.log_std_dev = layers.Dense(action_dim)

    def call(self, state, noisy=False):
        x = self.layer_2(self.layer_1(self.input_layer(state)))
        mean = self.mean(x)
        log_std = self.log_std_dev(x)
        log_std_clipped = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std_dev = tf.exp(log_std_clipped)
        policy = tfp.distributions.Normal(mean, std_dev)
        noise = tfp.distributions.Normal(0,1).sample()
        if DEBUG_ACTOR_NET:
            print("\n\n\t++++++++++++++++ DEBUG - ACTOR NET [ACTOR.CALL]++++++++++++++++\n")
            print("\t----------------------------input_actor----------------------------")
            print("\t", state)
            print("\t----------------------------mean----------------------------")
            print("\t", mean)
            print("\t----------------------------max-min mean----------------------------")
            print("\t", max(mean.numpy()[0]), "-",min(mean.numpy()[0]))
            print("\t----------------------------log_std----------------------------")
            print("\t", log_std)
            print("\t----------------------------max-min log_std----------------------------")
            print("\t", max(log_std.numpy()[0]), "-", min(log_std.numpy()[0]))
            print("\t----------------------------log_std_clipped----------------------------")
            print("\t", log_std_clipped)
            print("\t----------------------------noise----------------------------")
            print("\t", noise)
            a = input("\n\n\tPress Enter to continue...")
        if noisy:
            action_sample = mean + noise*std_dev
            if DEBUG_ACTOR_SAMPLE:
                print("\n\n\t++++++++++++++++ DEBUG - ACTOR SAMPLE [ACTOR.CALL]++++++++++++++++\n")
                print("\t----------------------------action noisy----------------------------")
        else:
            action_sample = policy.sample()
            if DEBUG_ACTOR_SAMPLE:
                print("\n\n\t++++++++++++++++ DEBUG - ACTOR SAMPLE [ACTOR.CALL]++++++++++++++++\n")
                print("\t----------------------------normal action----------------------------")
        squashed_actions = tf.tanh(action_sample)
        logprob = policy.log_prob(action_sample) - tf.math.log(1.0 - tf.pow(squashed_actions, 2) + NOISE)
        if DEBUG_ACTOR_SAMPLE:
            print("\t", action_sample)
            print("\t----------------------------squashed action----------------------------")
            print("\t", squashed_actions)
            print("\t----------------------------policy.log_prob----------------------------")
            print("\t", policy.log_prob(action_sample))
            print("\t----------------------------policy.log_prob after sum----------------------------")
            print("\t", tf.reduce_sum(policy.log_prob(action_sample), axis=1, keepdims=True))
            print("\t----------------------------tf.math.log(1.0 - tf.pow(squashed_actions, 2) + NOISE----------------------------")
            print("\t", tf.math.log(1.0 - squashed_actions**2 + NOISE))
            print("\t----------------------------// after sum----------------------------")
            print("\t", tf.reduce_sum(tf.math.log(1.0 - squashed_actions**2 + NOISE), axis=1, keepdims=True))
            print("\t----------------------------log prob before reduce_sum----------------------------")
            print("\t", logprob)
        logprob = tf.reduce_sum(logprob, axis=1, keepdims=True)
        if DEBUG_ACTOR_SAMPLE:           
            print("\t----------------------------final log_probs----------------------------")
            print("\t", logprob)
            a = input("\n\n\tPress Enter to continue...")
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
