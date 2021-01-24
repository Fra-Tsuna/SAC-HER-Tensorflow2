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

    
class ActorNetwork():
    def __init__(self, alpha, input_dims, max_action, l1_dims=256, l2_dims=256, n_actions=2):
        self.input_dims=input_dims
        self.max_action=max_action
        self.l1_dims=l1_dims
        self.l2_dims=l2_dims
        self.n_actions=n_actions
        self.noise_factor=1e-6

        self.model=Sequential()
        self.model.add(Input(*self.input_dims,))
        self.model.add(Dense(self.l1_dims))
        self.model.add(ReLU)
        self.model.add(Dense(self.l2_dims))
        self.model.add(ReLU)

        self.mu=self.model
        self.mu.add(Dense(self.n_actions))
        self.sigma=self.model
        self.sigma.add(Dense(self.n_actions))

        self.optimizer=Adam(learning_rate=alpha)
        self.mu.compile(optimizer=optimizer)
        self.sigma.compile(optimizer=optimizer)

    def forward(self, state):
        mu=self.mu.predict(state)
        sigma=self.sigma.predict(state)
        sigma=np.clip(sigma,self.noise_factor,1)

    def sample_normal(self, state, reparam=True):
        noise = tfd.Normal(0,1).sample()
        mu, sigma = forward(state)
        prob = tfd.Normal(mu, sigma)

        if reparam = True:
            actions = noise + prob.sample()
        else
            actions = prob.sample()
        
        action = tf.tanh(actions)
        log_probs=prob.log_prob(actions)
        log_probs -= np.log(1-(action)^2 + self.noise_factor)
        #bisogna fare la somma degli elementi, ma non capisco se 
        #log probs è una lista di elementi o una qualche altra 
        #struttura dati

        return action, log_probs
